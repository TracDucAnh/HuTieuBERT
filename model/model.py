import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers import RobertaConfig, PreTrainedModel
from .embeddings import BoundaryAwareEmbeddings
from .bias_utils import create_bias_matrix
from transformers.modeling_outputs import MaskedLMOutput
from transformers.modeling_outputs import SequenceClassifierOutput

class MorphemeAwareRobertaModel(RobertaModel):
    """
    PhoBERT mở rộng với:
    - BoundaryAwareEmbeddings (BMES + gate)
    - BMES bias hook trên attention head, hỗ trợ batch
    """

    def __init__(self, config, target_heads=None, alpha=0.1, beta=-0.05, gamma=0.0, delta=0.0, block_bmes_emb = False ,**kwargs):
        super().__init__(config, **kwargs)

        self.embeddings = BoundaryAwareEmbeddings(config, **kwargs)
        self.block_bmes_emb = block_bmes_emb

        # Bias params
        self.target_heads = target_heads or {}
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        self.tokenizer = None
        self.patched_forwards = {}
        self.bias_matrix = None

    def set_tokenizer(self, tokenizer):
        assert tokenizer is not None
        self.tokenizer = tokenizer

    def set_bias_matrix(self, bmes_tags):
        """
        bmes_tags: tensor [B, seq_len] hoặc [seq_len]
        Trả về tensor [B, num_heads, seq_len, seq_len]
        """
        if isinstance(bmes_tags, torch.Tensor) and bmes_tags.dim() == 1:
            bmes_tags = bmes_tags.unsqueeze(0)

        batch_size, seq_len = bmes_tags.shape
        bias_np = create_bias_matrix(bmes_tags, alpha=self.alpha, beta=self.beta, gamma=self.gamma, delta=self.delta)
        bias_tensor = torch.tensor(bias_np, dtype=torch.float32, device=next(self.parameters()).device)
        num_heads = self.config.num_attention_heads
        bias_tensor = bias_tensor.unsqueeze(1).repeat(1, num_heads, 1, 1)
        self.bias_matrix = bias_tensor

    def _create_patched_forward(self, layer_idx, head_indices, original_forward, attn_module):
        """
        Tạo forward function mới có cộng bias vào attention scores trước softmax
        """
        def patched_forward(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=None,
            output_attentions=False,
            **kwargs
        ):
            batch_size, seq_length = hidden_states.shape[:2]
            
            # Tính Q, K, V
            query_layer = attn_module.query(hidden_states)
            
            # Xử lý key và value
            is_cross_attention = encoder_hidden_states is not None
            
            if is_cross_attention:
                key_layer = attn_module.key(encoder_hidden_states)
                value_layer = attn_module.value(encoder_hidden_states)
            elif past_key_value is not None:
                key_layer = attn_module.key(hidden_states)
                value_layer = attn_module.value(hidden_states)
                key_layer = torch.cat([past_key_value[0], key_layer], dim=1)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=1)
            else:
                key_layer = attn_module.key(hidden_states)
                value_layer = attn_module.value(hidden_states)
            
            # Reshape để split heads
            def split_heads(tensor, num_heads, head_dim):
                new_shape = tensor.size()[:-1] + (num_heads, head_dim)
                tensor = tensor.view(new_shape)
                return tensor.permute(0, 2, 1, 3)
            
            num_heads = attn_module.num_attention_heads
            head_dim = attn_module.attention_head_size
            
            query_layer = split_heads(query_layer, num_heads, head_dim)
            key_layer = split_heads(key_layer, num_heads, head_dim)
            value_layer = split_heads(value_layer, num_heads, head_dim)
            
            if hasattr(attn_module, 'is_decoder') and attn_module.is_decoder:
                past_key_value = (key_layer, value_layer)
            
            # Tính attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / torch.sqrt(
                torch.tensor(head_dim, dtype=attention_scores.dtype, device=attention_scores.device)
            )
            
            # ✅ CỘNG BIAS VÀO ĐÂY - TRƯỚC SOFTMAX
            if self.bias_matrix is not None:
                # print("Adding bias matrix")
                B, H, L, _ = attention_scores.shape
                bias = self.bias_matrix
                
                if bias.size(0) != B:
                    bias = bias[:B]
                if bias.size(-1) != L:
                    bias = bias[:, :, :L, :L]
                
                for h in head_indices:
                    if h < H:
                        attention_scores[:, h, :, :] = attention_scores[:, h, :, :] + bias[:, h, :, :]
            
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            
            attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = attn_module.dropout(attention_probs)
            
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            
            # Tính context layer
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (attn_module.all_head_size,)
            context_layer = context_layer.view(new_context_layer_shape)
            
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            
            if hasattr(attn_module, 'is_decoder') and attn_module.is_decoder:
                outputs = outputs + (past_key_value,)
            
            return outputs
        
        return patched_forward

    def _patch_attention_layer(self, layer_idx, head_indices):
        """
        Monkey patch forward method của attention layer
        """
        attn_module = self.encoder.layer[layer_idx].attention.self
        
        if layer_idx not in self.patched_forwards:
            original_forward = attn_module.forward
            self.patched_forwards[layer_idx] = (attn_module, original_forward)
            
            patched_forward = self._create_patched_forward(
                layer_idx, head_indices, original_forward, attn_module
            )
            attn_module.forward = patched_forward

    def prepare_bias_patches(self):
        """
        Patch tất cả các layer có target heads
        """
        self.remove_bias_patches()
        for layer_idx, heads in self.target_heads.items():
            self._patch_attention_layer(layer_idx, heads)

    def remove_bias_patches(self):
        """
        Khôi phục lại original forward methods
        """
        for layer_idx, (attn_module, original_forward) in self.patched_forwards.items():
            attn_module.forward = original_forward
        self.patched_forwards = {}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        bmes_ids=None,
        bmes_tags=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values_length=0,  # Thêm param này cho embedding layer
    ):
        # Xử lý bmes_ids/bmes_tags
        if bmes_ids is None and bmes_tags is not None:
            bmes_ids = bmes_tags

        if inputs_embeds is None and self.block_bmes_emb == False:
            # print("Using bmes embeddings")
            inputs_embeds = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                bmes_ids=bmes_ids,  # ✅ Truyền bmes_ids vào embedding
                past_key_values_length=past_key_values_length
            )

        if self.block_bmes_emb == True:
            # print("Block bmes embeddings")
            inputs_embeds = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                bmes_ids=None,  # ✅ Không truyền BMES, chỉ dùng embedding gốc
                past_key_values_length=past_key_values_length
            )

        # Set bias matrix nếu có bmes_ids
        if bmes_ids is not None:
            self.set_bias_matrix(bmes_ids)

        # Patch attention layers nếu có target heads
        if self.target_heads:
            self.prepare_bias_patches()

        output_attentions = True if output_attentions is None else output_attentions

        # ✅ Gọi parent forward NHƯNG truyền inputs_embeds thay vì input_ids
        outputs = super().forward(
            input_ids=None,  # ✅ Set None vì đã có inputs_embeds
            attention_mask=attention_mask,
            token_type_ids=None,  # ✅ Set None vì đã được xử lý trong embedding
            position_ids=None,  # ✅ Set None vì đã được xử lý trong embedding
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,  # ✅ Dùng embedding đã tính
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Cleanup patches
        self.remove_bias_patches()
        return outputs
    

class MorphemeAwareRobertaForMaskedLM(PreTrainedModel):
    """
    HuTieuBert mở rộng cho Masked Language Modeling.
    Hỗ trợ bias attention theo BMES và tham số hóa alpha/beta/gamma.
    """
    config_class = RobertaConfig

    def __init__(
        self,
        config,
        target_heads=None,
        alpha=0.1,
        beta=-0.05,
        gamma=0.0,
        delta=0.0,
    ):
        super().__init__(config)

        # Truyền tham số xuống MorphemeAwareRobertaModel
        self.roberta = MorphemeAwareRobertaModel(
            config,
            target_heads=target_heads,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        # Head để dự đoán token bị che
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie weight: chia sẻ embedding giữa input và output
        self.tie_weights()
        self.init_weights()

    def tie_weights(self):
        self.lm_head.weight = self.roberta.embeddings.word_embeddings.weight

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        bmes_ids=None,
        bmes_tags=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Forward qua Roberta backbone có BMES bias
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            bmes_ids=bmes_ids,
            bmes_tags=bmes_tags,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        prediction_scores = self.lm_head(sequence_output)

        # Tính loss nếu có label
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size),
                labels.view(-1)
            )

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutput(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

class MorphemeAwareRobertaForSequenceClassification(PreTrainedModel):
    """
    HuTieuBert cho classification tasks.
    Sử dụng MorphemeAwareRobertaModel làm encoder + classification head.
    """
    config_class = RobertaConfig

    def __init__(
        self,
        config,
        num_labels=2,
        target_heads=None,
        alpha=0.1,
        beta=-0.05,
        gamma=0.0,
        delta=0.0,
    ):
        super().__init__(config)
        self.num_labels = num_labels
        self.config = config

        self.roberta = MorphemeAwareRobertaModel(
            config,
            target_heads=target_heads,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(config.hidden_dropout_prob),
            nn.Linear(config.hidden_size, num_labels)
        )

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        bmes_ids=None,
        bmes_tags=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            bmes_ids=bmes_ids,
            bmes_tags=bmes_tags,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]
        cls_output = sequence_output[:, 0, :]  # [batch_size, hidden_size]

        logits = self.classifier(cls_output)  # [batch_size, num_labels]

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                # Regression
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())
            else:
                # Classification
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )