import logging
from typing import Optional, Tuple
import re
from transformers import PreTrainedTokenizerFast

class SpanExtractor:
    """Extracts start and end indices for target phrases in Vietnamese text"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
      
    def get_span_indices(self, text: str, target_phrase: str) -> Optional[Tuple[int, int]]:
        """
        Finds start and end token indices for target phrase
        Handles Vietnamese subword tokenization with case-insensitive matching

        Args:
            text: Full input text
            target_phrase: Target phrase to locate

        Returns:
            Tuple (start_idx, end_idx) if found, else None
        """
        # Normalize whitespace for better matching
        text = ' '.join(text.split())
        target_phrase = ' '.join(target_phrase.split())

        if not target_phrase:
            self.logger.warning("Target phrase is empty")
            return None

        # Tokenize text first to get detailed debugging info
        encoding = self.tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=256 ,
            add_special_tokens=True
        )

        # Debug: print tokenization details
        tokens = self.tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        print(f"Debug - Text: {text}")
        print(f"Debug - Target: {target_phrase}")
        print(f"Debug - Tokens: {tokens}")
        print(f"Debug - Offsets: {encoding['offset_mapping']}")

        # Use case-insensitive search for character positions
        normalized_text = text.lower()
        normalized_target = target_phrase.lower()

        # Find all possible matches
        matches = []
        start = 0
        target_len = len(normalized_target)
        while start <= len(normalized_text) - target_len:
            pos = normalized_text.find(normalized_target, start)
            if pos == -1:
                break

            # Check if this is a word boundary match for short phrases
            before_ok = (pos == 0) or (not normalized_text[pos-1].isalnum())

            after_ok = (pos + target_len == len(normalized_text)) or \
                                  (not normalized_text[pos + target_len].isalnum())

            if before_ok and after_ok:
                matches.append((pos, pos + len(target_phrase)))


            start = pos + 1

        if not matches:
            self.logger.warning(f"Could not find token indices for '{target_phrase}' in '{text}'")
            return None

        # Try each match to find the best token alignment
        offsets = encoding["offset_mapping"]
        max_token_idx = len(offsets) - 1

        for start_char, end_char in matches:
            print(f"Debug - Trying match at chars {start_char}-{end_char}: '{text[start_char:end_char]}'")

            start_idx, end_idx = None, None

            for i, (char_start, char_end) in enumerate(offsets):
                # Skip special tokens with (0, 0) offsets
                if char_start == 0 and char_end == 0 and i > 0:
                    continue

                # Check for overlap with target span
                if char_end <= start_char:  # Token ends before target
                    continue
                if char_start >= end_char:  # Token starts after target
                    break

                # Token overlaps with target span
                if start_idx is None:
                    start_idx = i
                end_idx = i

            max_length = self.tokenizer.model_max_length - 2  
            
            if start_idx is not None and end_idx is not None:
                print(f"Debug - Found token range: {start_idx}-{end_idx}")
                print(f"Debug - Corresponding tokens: {tokens[start_idx:end_idx+1]}")
                start_idx = min(start_idx, max_length)
                end_idx = min(end_idx, max_length)
                end_idx = max(start_idx, end_idx)
                return (start_idx, end_idx)

        self.logger.warning(f"Could not find token indices for '{target_phrase}' in '{text}'")
        return None
    
    def get_span_text_from_indices(self, text, span_indices):
        """Lấy văn bản tương ứng với chỉ số span - phiên bản cải thiện"""
        if span_indices is None:
            return None
        encoding = self.tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        start_idx, end_idx = span_indices

        if start_idx >= len(encoding["offset_mapping"]) or end_idx >= len(encoding["offset_mapping"]):
            return None

        max_idx = len(encoding["offset_mapping"]) - 1
        if start_idx > max_idx or end_idx > max_idx or start_idx < 0 or end_idx < 0:
            self.logger.warning(f"Invalid span indices: ({start_idx}, {end_idx}), max_idx: {max_idx}")
            return None
        
        start_char = encoding["offset_mapping"][start_idx][0]
        end_char = encoding["offset_mapping"][end_idx][1]

        span_text = text[start_char:end_char]

        # Trích xuất văn bản trực tiếp từ vị trí ký tự
        span_text = text[start_char:end_char]

        return span_text
    
class SentenceMasking:
    def __init__(self,tokenizer: PreTrainedTokenizerFast):
        self.tokenizer=tokenizer 
        
    def create_masked_version(self,
        text: str, 
        target_phrase: str, 
        
    ) -> Tuple[Optional[str], Optional[Tuple[int, int]]]:
        mask_token = self.tokenizer.mask_token

        text = ' '.join(text.split())
        target_phrase = ' '.join(target_phrase.split())
        
        if not target_phrase or not text:
            return None, None
        
        extractor = SpanExtractor(self.tokenizer)
        
        span_indices = extractor.get_span_indices(text, target_phrase)
        
        if span_indices is None:
            return None, None
        
        encoding = self.tokenizer(text, 
                                  add_special_tokens=True, 
                                  return_offsets_mapping=True,
                                  truncation=True)
        input_ids = encoding["input_ids"]
        
        if not input_ids:
            return None, None
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        if not tokens or any(tok is None for tok in tokens):
            return None, None
        
        masked_tokens = tokens.copy()
        start_idx, end_idx = span_indices
        
        n_tokens = len(tokens)
        start_idx = max(0, min(start_idx, n_tokens - 1))
        end_idx = max(start_idx, min(end_idx, n_tokens - 1))

        masked_tokens[start_idx:end_idx+1] = [mask_token+"</w>"]
        
        filtered_tokens = [tok for tok in masked_tokens if tok is not None]
        
        try:
            masked_text = self.tokenizer.convert_tokens_to_string(filtered_tokens)
        except Exception as e:
            print(f"Error converting tokens to string: {e}")
            print(f"Tokens: {filtered_tokens}")
            return None, None
        masked_text = re.sub(r'\s*<[/]?s>\s*', ' ', masked_text).strip()
        return masked_text, (start_idx, end_idx)