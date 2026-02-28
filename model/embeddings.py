import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

class BoundaryAwareEmbeddings(RobertaEmbeddings):
    def __init__(self, config, adaptive=True, boundary_ratio=0.2, **kwargs):
        """
        config: RobertaConfig từ HuggingFace
        adaptive: nếu True dùng gating động
        boundary_ratio: nếu adaptive=False, tỉ lệ weight cố định cho boundary embedding
        kwargs: giữ để HuggingFace from_pretrained gọi được
        """
        super().__init__(config, **kwargs)  # HF sẽ load weight chuẩn của RobertaEmbeddings

        # Tham số custom
        self.adaptive = adaptive
        self.boundary_ratio = boundary_ratio

        # Lớp BMES embeddings mới
        self.bmes_embeddings = nn.Embedding(4, config.hidden_size)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        # LayerNorm riêng cho embedding fusion
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        bmes_ids=None,
        past_key_values_length=0
    ):
        # Embedding chuẩn từ Roberta
        E_token = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length
        )

        # Nếu có BMES ids, fusing với embedding gốc
        if bmes_ids is not None:
            E_boundary = self.bmes_embeddings(bmes_ids)

            if self.adaptive:
                # Gate động
                concat = torch.cat([E_boundary, E_token], dim=-1)
                W = self.sigmoid(self.gate(concat))
                E_fused = W * E_boundary + (1 - W) * E_token
            else:
                # Weight cố định
                alpha = self.boundary_ratio
                E_fused = alpha * E_boundary + (1 - alpha) * E_token

            embeddings = self.final_layernorm(E_fused)
        else:
            embeddings = E_token

        return embeddings
