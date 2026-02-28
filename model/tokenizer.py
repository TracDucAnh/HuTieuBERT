import torch
import re
import unicodedata
import py_vncorenlp
from transformers import AutoTokenizer

class MorphemeAwareTokenizer(AutoTokenizer):
    def __init__(self, pretrained_model_name="vinai/phobert-base", vncorenlp_dir='/content/vncorenlp', **kwargs):
        # Khởi tạo tokenizer HF gốc
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, **kwargs)

        # Khởi tạo VnCoreNLP cho word segmentation
        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(
            annotators=["wseg"],
            save_dir=vncorenlp_dir
        )

    def __len__(self):
        # Trả về vocab size
        return self.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, vncorenlp_dir='/content/vncorenlp', **kwargs):
        """
        Load tokenizer từ Hugging Face và giữ logic custom
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
    # ✅ Bổ sung để tương thích với DataCollator
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
        Cho phép DataCollatorForLanguageModeling sử dụng pad() như tokenizer Hugging Face.
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
        Trả về mask cho special tokens (1 = special token, 0 = normal token).
        Cần thiết cho DataCollatorForLanguageModeling.
        """
        return self.tokenizer.get_special_tokens_mask(
            token_ids_0=token_ids_0,
            token_ids_1=token_ids_1,
            already_has_special_tokens=already_has_special_tokens
        )

    def convert_tokens_to_ids(self, tokens):
        """
        Chuyển tokens thành IDs. Cần cho một số collator.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """
        Chuyển IDs thành tokens.
        """
        return self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=skip_special_tokens)

    def to_bmes(self, text):
        """
        Tạo danh sách (syllable, BMES-tag) từ text hoặc list[text].
        Nếu là list -> trả về list[list[(syllable, tag)]]
        Nếu là str  -> trả về list[(syllable, tag)]
        """
        if isinstance(text, list):
            return [self.to_bmes(t) for t in text]

        if not isinstance(text, str):
            text = str(text)

        segmented = self.rdrsegmenter.word_segment(text)
        
        # Trường hợp output là list nhiều câu → gộp lại theo từng câu riêng
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
        Align BMES tags với subwords, xử lý các trường hợp:
        - Dấu câu dính với chữ (vd: 'c.', '3.')
        - Ký tự đặc biệt, <unk> tokens
        - Subword splitting phức tạp
        
        🔧 FIX: Xử lý <unk> token bằng cách skip nó và tiếp tục alignment
        """
        aligned_tags = []
        syll_idx = 0
        buffer_raw = ""
        subword_positions = []
        
        i = 0
        while i < len(subwords_list):
            sub = subwords_list[i]
            
            # Special tokens - luôn tag là 'S'
            if sub in ["<s>", "</s>", "<pad>", "<mask>"]:
                aligned_tags.append("S")
                i += 1
                continue
            
            # 🔧 XỬ LÝ <unk> TOKEN
            if sub == "<unk>":
                # <unk> token là biểu diễn của 1 ký tự không được vocab nhận diện
                # Gán tag 'S' cho nó và bỏ qua 1 syllable trong bmes_list nếu có
                aligned_tags.append("S")
                
                # Nếu còn syllable, skip nó vì đã được thay thế bằng <unk>
                if syll_idx < len(bmes_list):
                    syll_idx += 1
                
                # Reset buffer để tránh cascade errors
                buffer_raw = ""
                subword_positions = []
                
                i += 1
                continue
            
            # Hết syllables - tag còn lại là 'S'
            if syll_idx >= len(bmes_list):
                aligned_tags.append("S")
                i += 1
                continue
            
            # Lấy syllable hiện tại
            syll, tag = bmes_list[syll_idx]
            clean_sub = sub.replace("▁", "").replace("@@", "")
            
            # Normalize để so sánh
            normalized_syll = self.normalize_text(syll)
            
            # Case 1: Syllable là dấu câu thuần túy
            if self.is_punctuation(syll):
                # Kiểm tra xem subword có chứa dấu câu này không
                if clean_sub == syll or syll in clean_sub:
                    aligned_tags.append("S")
                    syll_idx += 1
                    i += 1
                    # Reset buffer nếu đang xử lý
                    buffer_raw = ""
                    subword_positions = []
                    continue
            
            # Case 2: Subword có dấu câu dính (vd: 'c.', 'i.')
            # Tách phần chữ và dấu câu
            word_part = ""
            punct_part = ""
            
            # Pattern để tách: chữ cái/số ở đầu, dấu câu ở cuối
            match = re.match(r'^([a-zA-ZÀ-ỹ0-9]+)([^\w]+)$', clean_sub, re.UNICODE)
            if match:
                word_part = match.group(1)
                punct_part = match.group(2)
            else:
                word_part = clean_sub
                punct_part = ""
            
            # Xử lý phần chữ
            if word_part:
                buffer_raw += word_part
                subword_positions.append(len(aligned_tags))
                aligned_tags.append(tag)  # Tag tạm thời
                
                normalized_buffer = self.normalize_text(buffer_raw)
                
                # Kiểm tra buffer có khớp với syllable chưa
                if normalized_buffer == normalized_syll:
                    # Gán lại tags đúng cho tất cả subwords trong buffer
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
                    
                    # Reset buffer và tăng syllable index
                    buffer_raw = ""
                    subword_positions = []
                    syll_idx += 1
                    
                    # Xử lý phần dấu câu nếu có
                    if punct_part:
                        # Kiểm tra syllable tiếp theo có phải dấu câu không
                        if syll_idx < len(bmes_list):
                            next_syll, next_tag = bmes_list[syll_idx]
                            if self.is_punctuation(next_syll) or next_syll == punct_part:
                                # Dấu câu này thuộc syllable tiếp theo, không thêm tag
                                syll_idx += 1
            
            i += 1
        
        return aligned_tags

    def __call__(self, text, **kwargs):
        # Nếu là list → xử lý batch
        if isinstance(text, list):
            # 1. Tokenize cả batch bằng tokenizer gốc
            encoded = self.tokenizer(
                text,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                return_tensors=kwargs.get("return_tensors", None),
            )

            # 2. Tạo BMES tags cho từng câu
            BMES_MAP = {"B": 0, "M": 1, "E": 2, "S": 3}
            bmes_tags_list = []
            for i, t in enumerate(text):
                bmes_list = self.to_bmes(t)
                subwords = self.tokenizer.convert_ids_to_tokens(encoded["input_ids"][i].tolist())
                bmes_tags = self.align_bmes_to_subwords(bmes_list, subwords)
                
                # Chuyển sang tensor nếu cần
                if kwargs.get("return_tensors") == "pt":
                    bmes_tags = torch.tensor([BMES_MAP[tag] for tag in bmes_tags])
                bmes_tags_list.append(bmes_tags)

            # Padding BMES tags giống input_ids
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

        # Nếu là string đơn → xử lý như cũ
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