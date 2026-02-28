import torch
import numpy as np

def create_bias_matrix(bmes_tags, alpha=0.1, beta=-0.05, gamma=0.0, delta=0.0):
    """
    Hỗ trợ:
    - bmes_tags: shape [seq_len] (1 sample) hoặc [B, seq_len] (batch)
    Trả về bias_matrix:
    - 1 sample: [seq_len, seq_len]
    - batch: [B, seq_len, seq_len]
    """
    def single_bias(seq_tags):
        # Chuyển tensor -> list ['B','M','E','S']
        if isinstance(seq_tags, torch.Tensor):
            BMES_MAP_INV = {0:'B',1:'M',2:'E',3:'S'}
            seq_tags = [BMES_MAP_INV[t.item()] if isinstance(t, torch.Tensor) else BMES_MAP_INV[t] for t in seq_tags.tolist()]

        seq_len = len(seq_tags)
        bias_matrix = np.zeros((seq_len, seq_len))

        # Nhóm token theo từ
        word_groups = []
        current_group = [0]
        for i in range(1, seq_len):
            prev_tag = seq_tags[i-1]
            curr_tag = seq_tags[i]
            if prev_tag in ['E','S']:
                word_groups.append(current_group)
                current_group = [i]
            else:
                current_group.append(i)
        if current_group:
            word_groups.append(current_group)

        # Điền bias
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
        # batch
        batch_bias = [single_bias(bmes_tags[i]) for i in range(bmes_tags.size(0))]
        return np.stack(batch_bias, axis=0)  # [B, seq_len, seq_len]
    else:
        # 1 sample
        return single_bias(bmes_tags)  # [seq_len, seq_len]