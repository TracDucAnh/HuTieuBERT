import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaEmbeddings

class BoundaryAwareEmbeddings(RobertaEmbeddings):
    def __init__(self, config, adaptive=True, boundary_ratio=0.2, **kwargs):
        """
        config: RobertaConfig instance from Hugging Face
        adaptive: if True, use dynamic gating
        boundary_ratio: fixed interpolation weight for boundary embeddings
            when adaptive=False
        kwargs: preserved for compatibility with Hugging Face from_pretrained
        """
        super().__init__(config, **kwargs)  # Hugging Face loads the base RobertaEmbeddings weights

        # Custom configuration
        self.adaptive = adaptive
        self.boundary_ratio = boundary_ratio

        # Additional BMES embedding layers
        self.bmes_embeddings = nn.Embedding(4, config.hidden_size)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.sigmoid = nn.Sigmoid()

        # Dedicated LayerNorm for fused embeddings
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
        # Standard RoBERTa embeddings
        E_token = super().forward(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length
        )

        # Fuse BMES embeddings with the base embeddings when BMES ids are available
        if bmes_ids is not None:
            E_boundary = self.bmes_embeddings(bmes_ids)

            if self.adaptive:
                # Dynamic gating
                concat = torch.cat([E_boundary, E_token], dim=-1)
                W = self.sigmoid(self.gate(concat))
                E_fused = W * E_boundary + (1 - W) * E_token
            else:
                # Fixed interpolation weight
                alpha = self.boundary_ratio
                E_fused = alpha * E_boundary + (1 - alpha) * E_token

            embeddings = self.final_layernorm(E_fused)
        else:
            embeddings = E_token

        return embeddings
