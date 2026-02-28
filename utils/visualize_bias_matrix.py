import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np

def visualize_bias_matrix(bias_matrix, encoded=None, tokenizer=None, tokens=None, title="Bias Matrix Visualization"):
    """
    Args:
        bias_matrix: torch.Tensor, shape [seq_len, seq_len] hoặc [1, num_heads, seq_len, seq_len]
        encoded: dict từ tokenizer, chứa 'input_ids' (tùy chọn)
        tokenizer: tokenizer dùng để convert input_ids sang token (tùy chọn)
        tokens: list[str], nhãn token nếu muốn tự truyền
        title: tiêu đề heatmap
    """
    # Nếu bias_matrix là 4D -> [1, num_heads, seq_len, seq_len]
    if isinstance(bias_matrix, torch.Tensor):
        bias_matrix = bias_matrix.detach().cpu()
        if bias_matrix.ndim == 4:
            # trung bình trên head
            bias_matrix = bias_matrix.mean(dim=1).squeeze(0)
        elif bias_matrix.ndim == 3:
            bias_matrix = bias_matrix.squeeze(0)
        bias_matrix = bias_matrix.numpy()
    
    seq_len = bias_matrix.shape[0]

    # Lấy tokens từ input_ids nếu chưa có
    if tokens is None:
        if encoded is not None and tokenizer is not None:
            input_ids = encoded.get("input_ids")
            if isinstance(input_ids, torch.Tensor):
                if input_ids.ndim == 2:  # batch
                    input_ids = input_ids[0]
                input_ids = input_ids.detach().cpu().tolist()
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
        else:
            tokens = [str(i) for i in range(seq_len)]
    else:
        tokens = tokens[:seq_len]

    # Vẽ heatmap
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
