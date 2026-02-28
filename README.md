# HuTieuBERT: Morpheme-Aware Transformer for Vietnamese

This repository implements a morpheme-aware Transformer architecture that enhances pretrained encoders with explicit morphological structure for isolating languages. 

By introducing two lightweight inductive biases:

- Adaptive Boundary-Token Fusion 

- Morpheme-Aware Attention Bias.

The model effectively captures compound cohesion and morpheme boundaries that standard Transformers often overlook. While optimized for Vietnamese.

![Architecture](/figures/ReDrawHuTieuBERT.png)


The design is portable to other isolating languages like Mandarin Chinese, consistently improving performance on syntactic tasks such as POS tagging and Named Entity Recognition (NER).

## Adaptive Boundary-Token Fusion

**Subword Alignment:**

- Syncing Labels: This step aligns word-structure tags (BMES) with the smaller sub-units of text created during tokenization.

- Maintaining Structure: By expanding these tags, the model ensures that multi-syllable words keep their linguistic meaning even when broken into pieces.

![Subword Alignment](/figures/alignment.png)

**Adaptive Interpolation Layer**

- Blending Information: This module combines standard word data with specific "boundary" information that marks where words start and end.

- Smart Filtering: A "gate" automatically decides how much boundary information is needed for each word based on its context.

- Rich Representation: The result is a more complete digital representation of the text that respects the natural boundaries of the language.

![Adaptive Boundary-Token Fusion](/figures/embeddings.png)

## Morpheme-Aware Attention Bias.

This module guides the model's focus by injecting a fixed structural prior into the early self-attention layers. It ensures that the "attention" mass respects the natural boundaries of compounds rather than spreading too thin across unrelated words.

![Vie](/figures/multi_layer_attention_1-2_all.png)

The bias is controlled by a matrix using four key parameters to modulate relationship scores:

- Alpha ($\alpha$): Strengthens focus between tokens that belong to the same compound phrase.

- Beta ($\beta$): Penalizes or "mutes" attention between tokens that belong to different compounds.

- Gamma ($\gamma$): Highlights and adjusts the importance of single-word units.

- Delta ($\delta$): Controls the strength of a token's focus on itself (self-attention bias).

By reweighting these connections, the model maintains a stable internal geometry while gaining a clearer understanding of linguistic structure. This method not only work with Vietnamese but also other Isolating Languages like Mandarin Chinese, Thai...

![Chi](/figures/machinesebert_attention_1-2_all.png)

## Example Usage

```python
import torch
import torch.nn as nn
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaEncoder, RobertaLayer
from transformers import RobertaConfig

from model.tokenizer import MorphemeAwareTokenizer
from model.embeddings import BoundaryAwareEmbeddings
from model.model import MorphemeAwareRobertaModel, MorphemeAwareRobertaForSequenceClassification

tokenizer = MorphemeAwareTokenizer.from_pretrained(
    "ducanhdinh/HuTieuBert",
    vncorenlp_dir="/content/vncorenlp",
    return_tensors="pt"
)

config = RobertaConfig.from_pretrained("ducanhdinh/HuTieuBert")

# Applied Structural Bias Matrix to Layer 1 and 2 full 12 heads
target_heads = {
    1: list(range(config.num_attention_heads)),
    2: list(range(config.num_attention_heads)),
}

model = MorphemeAwareRobertaForSequenceClassification(
    config,
    num_labels=label_num,
    target_heads=target_heads,
    alpha=0.5,
    beta=-0.3,
    gamma=0.0,
    delta=0.0,
)

model.roberta = MorphemeAwareRobertaModel.from_pretrained(
    "ducanhdinh/HuTieuBert",
    target_heads=target_heads,
    alpha=0.5,
    beta=-0.3,
    gamma=0.0,
    delta=0.0,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```

## Acknowledgement

### Paper (Coming Soon)

The full paper describing our method and experimental results will be released soon. If you find this work useful, please consider citing our paper:

```bibtex
@article{,
  title   = {Coming Soon},
  author  = {Coming Soon},
  journal = {Coming Soon},
  year    = {Coming Soon},
}
```

### Use of External Segmentation

This work makes use of **VnCoreNLP** - a Vietnamese natural language processing toolkit.

Copyright (C) 2018-2019 VnCoreNLP  
This program is free software: you can redistribute it and/or modify  
it under the terms of the GNU General Public License as published by  
the Free Software Foundation, either version 3 of the License, or  
(at your option) any later version.

This program is distributed in the hope that it will be useful,  
but **WITHOUT ANY WARRANTY**; without even the implied warranty of  
**MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.**  
See the [GNU General Public License](https://www.gnu.org/licenses/) for more details.

Repository: [https://github.com/vncorenlp/VnCoreNLP](https://github.com/vncorenlp/VnCoreNLP)