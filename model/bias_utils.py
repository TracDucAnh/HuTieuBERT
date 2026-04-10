import torch
import numpy as np

def create_bias_matrix(bmes_tags, alpha=0.1, beta=-0.05, gamma=0.0, delta=0.0):
    """
    Supports:
    - bmes_tags with shape [seq_len] for a single sample or [B, seq_len] for a batch

    Returns:
    - Single sample: [seq_len, seq_len]
    - Batch: [B, seq_len, seq_len]
    """
    def single_bias(seq_tags):
        # Convert tensor values to a BMES tag list such as ['B', 'M', 'E', 'S']
        if isinstance(seq_tags, torch.Tensor):
            BMES_MAP_INV = {0:'B',1:'M',2:'E',3:'S'}
            seq_tags = [BMES_MAP_INV[t.item()] if isinstance(t, torch.Tensor) else BMES_MAP_INV[t] for t in seq_tags.tolist()]

        seq_len = len(seq_tags)
        bias_matrix = np.zeros((seq_len, seq_len))

        # Group tokens by word boundaries
        word_groups = []
        current_group = [0]
        for i in range(1, seq_len):
            prev_tag = seq_tags[i-1]
            if prev_tag in ['E','S']:
                word_groups.append(current_group)
                current_group = [i]
            else:
                current_group.append(i)
        if current_group:
            word_groups.append(current_group)

        # Populate the bias matrix
        for i in range(seq_len):
            for j in range(seq_len):
                if i == j:
                    bias_matrix[i,j] = delta
                elif seq_tags[i]=='S' or seq_tags[j]=='S':
                    bias_matrix[i,j] = gamma
                else:
                    same_word = any(i in g and j in g for g in word_groups)
                    bias_matrix[i,j] = alpha if same_word else beta
        return bias_matrix

    if isinstance(bmes_tags, torch.Tensor) and bmes_tags.dim() == 2:
        # Batch input
        batch_bias = [single_bias(bmes_tags[i]) for i in range(bmes_tags.size(0))]
        return np.stack(batch_bias, axis=0)  # [B, seq_len, seq_len]
    else:
        # Single sample
        return single_bias(bmes_tags)  # [seq_len, seq_len]
