import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def visualize_bias_matrix(bias_matrix, encoded=None, tokenizer=None, tokens=None, title="Bias Matrix Visualization"):
    """
    Args:
        bias_matrix: torch.Tensor with shape [seq_len, seq_len] or
            [1, num_heads, seq_len, seq_len]
        encoded: tokenizer output dict containing 'input_ids' (optional)
        tokenizer: tokenizer used to convert input_ids back to tokens (optional)
        tokens: list[str], custom token labels if provided directly
        title: heatmap title
    """
    # If bias_matrix is 4D, interpret it as [1, num_heads, seq_len, seq_len]
    if isinstance(bias_matrix, torch.Tensor):
        bias_matrix = bias_matrix.detach().cpu()
        if bias_matrix.ndim == 4:
            # Average across attention heads
            bias_matrix = bias_matrix.mean(dim=1).squeeze(0)
        elif bias_matrix.ndim == 3:
            bias_matrix = bias_matrix.squeeze(0)
        bias_matrix = bias_matrix.numpy()
    
    seq_len = bias_matrix.shape[0]

    # Recover tokens from input_ids when they are not explicitly provided
    if tokens is None:
        if encoded is not None and tokenizer is not None:
            input_ids = encoded.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                if input_ids.ndim == 2:  # Batch case
                    input_ids = input_ids[0]
                input_ids = input_ids.detach().cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
        else:
            tokens = [str(i) for i in range(seq_len)]
    else:
        tokens = tokens[:seq_len]

    # Draw the heatmap
    plt.figure(figsize=(max(6, seq_len * 0.6), max(6, seq_len * 0.6)))
    sns.heatmap(
        bias_matrix,
        cmap="RdYlGn",
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Bias Value"},
        xticklabels=tokens,
        yticklabels=tokens
    )
    plt.title(title, fontsize=14, fontweight="bold", pad=12)
    plt.xlabel("Key Tokens (j)", fontsize=11, fontweight="bold")
    plt.ylabel("Query Tokens (i)", fontsize=11, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()
