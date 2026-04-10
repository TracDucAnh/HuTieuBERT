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
    PhoBERT extended with:
    - BoundaryAwareEmbeddings (BMES + gating)
    - A BMES-based attention bias hook with batch support
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
        bmes_tags: tensor with shape [B, seq_len] or [seq_len]
        Returns a tensor with shape [B, num_heads, seq_len, seq_len]
        """
        if isinstance(bmes_tags, torch.Tensor) and bmes_tags.dim() == 1:
            bmes_tags = bmes_tags.unsqueeze(0)

        bias_np = create_bias_matrix(bmes_tags, alpha=self.alpha, beta=self.beta, gamma=self.gamma, delta=self.delta)
        bias_tensor = torch.tensor(bias_np, dtype=torch.float32, device=next(self.parameters()).device)
        num_heads = self.config.num_attention_heads
        bias_tensor = bias_tensor.unsqueeze(1).repeat(1, num_heads, 1, 1)
        self.bias_matrix = bias_tensor

    def _create_patched_forward(self, head_indices, attn_module):
        """
        Create a patched forward function that adds the bias to attention
        scores before the softmax step.
        """
        def patched_forward(
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            past_key_value=None,
            output_attentions=False,
            **kwargs
        ):
            
            # Compute Q, K, and V
            query_layer = attn_module.query(hidden_states)
            
            # Process key and value tensors
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
            
            # Reshape tensors for multi-head attention
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
            
            # Compute attention scores
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / torch.sqrt(
                torch.tensor(head_dim, dtype=attention_scores.dtype, device=attention_scores.device)
            )
            
            # Add the structural bias before applying softmax
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
            
            # Compute the context layer
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
        Monkey-patch the forward method of the attention layer.
        """
        attn_module = self.encoder.layer[layer_idx].attention.self
        
        if layer_idx not in self.patched_forwards:
            original_forward = attn_module.forward
            self.patched_forwards[layer_idx] = (attn_module, original_forward)
            
            patched_forward = self._create_patched_forward(
                head_indices, attn_module
            )
            attn_module.forward = patched_forward

    def prepare_bias_patches(self):
        """
        Patch all layers that contain target heads.
        """
        self.remove_bias_patches()
        for layer_idx, heads in self.target_heads.items():
            self._patch_attention_layer(layer_idx, heads)

    def remove_bias_patches(self):
        """
        Restore the original forward methods.
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
        past_key_values_length=0,  # Extra parameter required by the embedding layer
    ):
        # Normalize bmes_ids / bmes_tags inputs
        if bmes_ids is None and bmes_tags is not None:
            bmes_ids = bmes_tags

        if inputs_embeds is None and self.block_bmes_emb == False:
            # print("Using bmes embeddings")
            inputs_embeds = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                bmes_ids=bmes_ids,  # Pass BMES ids into the embedding layer
                past_key_values_length=past_key_values_length
            )

        if self.block_bmes_emb == True:
            # print("Block bmes embeddings")
            inputs_embeds = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                bmes_ids=None,  # Disable BMES and use only the base embeddings
                past_key_values_length=past_key_values_length
            )

        # Set the bias matrix when BMES ids are available
        if bmes_ids is not None:
            self.set_bias_matrix(bmes_ids)

        # Patch attention layers when target heads are specified
        if self.target_heads:
            self.prepare_bias_patches()

        output_attentions = True if output_attentions is None else output_attentions

        # Call the parent forward method using inputs_embeds instead of input_ids
        outputs = super().forward(
            input_ids=None,  # Use None because inputs_embeds is already provided
            attention_mask=attention_mask,
            token_type_ids=None,  # Use None because token types are handled in embeddings
            position_ids=None,  # Use None because positions are handled in embeddings
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,  # Reuse the precomputed embeddings
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Cleanup patches
        self.remove_bias_patches()
        return outputs
    

class MorphemeAwareRobertaForMaskedLM(PreTrainedModel):
    """
    HuTieuBERT extension for Masked Language Modeling.
    Supports BMES-guided attention bias with configurable alpha, beta, gamma,
    and delta parameters.
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

        # Pass the configuration parameters to MorphemeAwareRobertaModel
        self.roberta = MorphemeAwareRobertaModel(
            config,
            target_heads=target_heads,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
        )

        # Prediction head for masked tokens
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Tie input and output embedding weights
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
        # Forward pass through the RoBERTa backbone with BMES bias
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

        # Compute the loss when labels are provided
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
    HuTieuBERT for classification tasks.
    Uses MorphemeAwareRobertaModel as the encoder with a classification head.
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
