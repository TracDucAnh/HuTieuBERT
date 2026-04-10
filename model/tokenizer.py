import torch
import re
import unicodedata
import py_vncorenlp
from transformers import AutoTokenizer

class MorphemeAwareTokenizer(AutoTokenizer):
    def __init__(self, pretrained_model_name="vinai/phobert-base", vncorenlp_dir='/content/vncorenlp', **kwargs):
        # Initialize the base Hugging Face tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, **kwargs)

        # Initialize VnCoreNLP for word segmentation
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=vncorenlp_dir
        )

    def __len__(self):
        # Return the vocabulary size
        return self.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, vncorenlp_dir='/content/vncorenlp', **kwargs):
        """
        Load the tokenizer from Hugging Face while preserving the custom logic.
        """
        return cls(pretrained_model_name_or_path, vncorenlp_dir=vncorenlp_dir, **kwargs)
    

    @property
    def mask_token(self):
        return self.tokenizer.mask_token

    @property
    def pad_token(self):
        return self.tokenizer.pad_token

    @property
    def cls_token(self):
        return self.tokenizer.cls_token

    @property
    def sep_token(self):
        return self.tokenizer.sep_token

    @property
    def unk_token(self):
        return self.tokenizer.unk_token

    # =============================
    # Additional methods for DataCollator compatibility
    # =============================

    @property
    def mask_token_id(self):
        return self.tokenizer.mask_token_id

    @property
    def pad_token_id(self):
        return self.tokenizer.pad_token_id

    @property
    def cls_token_id(self):
        return self.tokenizer.cls_token_id

    @property
    def sep_token_id(self):
        return self.tokenizer.sep_token_id

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    def pad(self, encoded_inputs, padding=True, max_length=None, return_tensors=None, **kwargs):
        """
        Allow DataCollatorForLanguageModeling to call pad() as with a standard
        Hugging Face tokenizer.
        """
        return self.tokenizer.pad(
            encoded_inputs,
            padding=padding,
            max_length=max_length,
            return_tensors=return_tensors,
            **kwargs
        )

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Return a mask for special tokens (1 = special token, 0 = normal token).
        This is required by DataCollatorForLanguageModeling.
        """
        return self.tokenizer.get_special_tokens_mask(
            token_ids_0=token_ids_0,
            token_ids_1=token_ids_1,
            already_has_special_tokens=already_has_special_tokens
        )

    def convert_tokens_to_ids(self, tokens):
        """
        Convert tokens to IDs. Required by some collators.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Convert IDs back to tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def to_bmes(self, text):
        """
        Create a list of (syllable, BMES-tag) pairs from text or list[text].
        If the input is a list, return list[list[(syllable, tag)]].
        If the input is a string, return list[(syllable, tag)].
        """
        if isinstance(text, list):
            return [self.to_bmes(t) for t in text]

        if not isinstance(text, str):
            text = str(text)

        segmented = self.rdrsegmenter.word_segment(text)
        
        # If the segmenter returns multiple sentences, process them separately
        if isinstance(segmented, list):
            sentences = segmented
        else:
            sentences = [segmented]

        bmes_list = []

        for sent in sentences:
            words = sent.split()
            for word in words:
                sylls = word.split("_")
                n = len(sylls)
                if n == 1:
                    bmes_list.append((sylls[0], 'S'))
                else:
                    bmes_list.append((sylls[0], 'B'))
                    for mid in sylls[1:-1]:
                        bmes_list.append((mid, 'M'))
                    bmes_list.append((sylls[-1], 'E'))
        
        return bmes_list


    def normalize_text(self, text):
        text = text.replace("@@", "").replace("▁", "").strip()
        text = unicodedata.normalize('NFD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])
        text = re.sub(r'[^\w\s]', '', text)
        return text.lower()

    def is_punctuation(self, text):
        normalized = re.sub(r'[^\w\s]', '', text).strip()
        return normalized == ""

    def align_bmes_to_subwords(self, bmes_list, subwords_list):
        """
        Align BMES tags with subwords while handling:
        - Punctuation attached to text (for example: 'c.', '3.')
        - Special characters and <unk> tokens
        - Complex subword splits

        Fix: handle <unk> tokens by skipping them and continuing alignment.
        """
        aligned_tags = []
        syll_idx = 0
        buffer_raw = ""
        subword_positions = []
        
        i = 0
        while i < len(subwords_list):
            sub = subwords_list[i]
            
            # Special tokens always receive the 'S' tag
            if sub in ["<s>", "</s>", "<pad>", "<mask>"]:
                aligned_tags.append("S")
                i += 1
                continue
            
            # Handle <unk> tokens
            if sub == "<unk>":
                # <unk> represents a character not recognized by the vocabulary.
                # Assign the 'S' tag and skip one syllable in bmes_list if available.
                aligned_tags.append("S")
                
                # Skip the corresponding syllable because it was replaced by <unk>
                if syll_idx < len(bmes_list):
                    syll_idx += 1
                
                # Reset the buffer to prevent cascading alignment errors
                buffer_raw = ""
                subword_positions = []
                
                i += 1
                continue
            
            # If no syllables remain, assign 'S' to the rest
            if syll_idx >= len(bmes_list):
                aligned_tags.append("S")
                i += 1
                continue
            
            # Read the current syllable
            syll, tag = bmes_list[syll_idx]
            clean_sub = sub.replace("▁", "").replace("@@", "")
            
            # Normalize for comparison
            normalized_syll = self.normalize_text(syll)
            
            # Case 1: the syllable is pure punctuation
            if self.is_punctuation(syll):
                # Check whether the subword contains this punctuation mark
                if clean_sub == syll or syll in clean_sub:
                    aligned_tags.append("S")
                    syll_idx += 1
                    i += 1
                    # Reset the buffer if alignment was in progress
                    buffer_raw = ""
                    subword_positions = []
                    continue
            
            # Case 2: the subword contains trailing punctuation (for example: 'c.', 'i.')
            # Split the lexical part and the punctuation part
            word_part = ""
            punct_part = ""
            
            # Match a leading alphanumeric span followed by trailing punctuation
            match = re.match(r'^([a-zA-ZÀ-ỹ0-9]+)([^\w]+)$', clean_sub, re.UNICODE)
            if match:
                word_part = match.group(1)
                punct_part = match.group(2)
            else:
                word_part = clean_sub
                punct_part = ""
            
            # Process the lexical part
            if word_part:
                buffer_raw += word_part
                subword_positions.append(len(aligned_tags))
                aligned_tags.append(tag)  # Temporary tag assignment
                
                normalized_buffer = self.normalize_text(buffer_raw)
                
                # Check whether the buffer now matches the target syllable
                if normalized_buffer == normalized_syll:
                    # Reassign the correct tags to all subwords in the buffer
                    n = len(subword_positions)
                    if n > 1:
                        if tag == 'B':
                            aligned_tags[subword_positions[0]] = 'B'
                            for pos in subword_positions[1:]:
                                aligned_tags[pos] = 'M'
                        elif tag == 'E':
                            for pos in subword_positions[:-1]:
                                aligned_tags[pos] = 'M'
                            aligned_tags[subword_positions[-1]] = 'E'
                        elif tag == 'M':
                            for pos in subword_positions:
                                aligned_tags[pos] = 'M'
                        elif tag == 'S':
                            for pos in subword_positions:
                                aligned_tags[pos] = 'S'
                    else:
                        aligned_tags[subword_positions[0]] = tag
                    
                    # Reset the buffer and advance the syllable pointer
                    buffer_raw = ""
                    subword_positions = []
                    syll_idx += 1
                    
                    # Process the punctuation part if present
                    if punct_part:
                        # Check whether the next syllable is punctuation
                        if syll_idx < len(bmes_list):
                            next_syll, next_tag = bmes_list[syll_idx]
                            if self.is_punctuation(next_syll) or next_syll == punct_part:
                                # This punctuation belongs to the next syllable, so do not add a new tag
                                syll_idx += 1
            
            i += 1
        
        return aligned_tags

    def __call__(self, text, **kwargs):
        # If the input is a list, process it as a batch
        if isinstance(text, list):
            # 1. Tokenize the full batch with the base tokenizer
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors=kwargs.get("return_tensors", None),
            )

            # 2. Generate BMES tags for each sentence
            BMES_MAP = {"B": 0, "M": 1, "E": 2, "S": 3}
            bmes_tags_list = []
            for i, t in enumerate(text):
                bmes_list = self.to_bmes(t)
                subwords = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][i].tolist())
                bmes_tags = self.align_bmes_to_subwords(bmes_list, subwords)
                
                # Convert to a tensor when requested
                if kwargs.get("return_tensors") == "pt":
                    bmes_tags = torch.tensor([BMES_MAP[tag] for tag in bmes_tags])
                bmes_tags_list.append(bmes_tags)

            # Pad BMES tags to match the input_ids length
            if kwargs.get("return_tensors") == "pt":
                max_len = encoded["input_ids"].shape[1]
                padded_bmes = []
                for tags in bmes_tags_list:
                    pad_len = max_len - tags.shape[0]
                    if pad_len > 0:
                        tags = torch.cat([tags, torch.full((pad_len,), BMES_MAP["S"])])
                    padded_bmes.append(tags)
                encoded["bmes_tags"] = torch.stack(padded_bmes)
            else:
                encoded["bmes_tags"] = bmes_tags_list

            return encoded

        # If the input is a single string, use the single-example path
        bmes_list = self.to_bmes(text)
        encoded = self.tokenizer(text, add_special_tokens=True, **kwargs)

        input_ids = encoded["input_ids"]
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.squeeze(0).tolist()
        elif isinstance(input_ids[0], list):
            input_ids = input_ids[0]

        subwords = self.tokenizer.convert_ids_to_tokens(input_ids)
        bmes_tags = self.align_bmes_to_subwords(bmes_list, subwords)

        if kwargs.get("return_tensors") == "pt":
            BMES_MAP = {"B": 0, "M": 1, "E": 2, "S": 3}
            bmes_tags = torch.tensor([BMES_MAP[t] for t in bmes_tags]).unsqueeze(0)

        encoded['bmes_tags'] = bmes_tags
        return encoded
    
    def save_pretrained(self, save_directory, **kwargs):
        return self.tokenizer.save_pretrained(save_directory, **kwargs)
