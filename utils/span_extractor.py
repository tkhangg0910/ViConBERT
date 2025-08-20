import logging
from typing import Optional, Tuple
import re
from transformers import PreTrainedTokenizerFast
import torch
class SpanExtractor:
    """Extracts start and end indices for target phrases in Vietnamese text"""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
      
    def get_span_indices(self, text: str,
                         target_phrase: str,
                         debug: bool=False) -> Optional[Tuple[int, int]]:
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
        if debug:
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
            if debug:
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
                if debug:
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
    
class TargetWordMaskExtractor:
    def __init__(self, tokenizer):
        """
        Initialize the mask extractor with a tokenizer.
        
        Args:
            tokenizer: HuggingFace tokenizer (e.g., PhoBERT tokenizer)
        """
        self.tokenizer = tokenizer
    
    def extract_target_mask(self, sentence: str, target_word: str, case_sensitive: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extract mask for target word in sentence.
        
        Args:
            sentence: Input sentence containing the target word
            target_word: The target word to mask
            case_sensitive: Whether to match case sensitively
            
        Returns:
            Tuple of (input_ids, attention_mask, target_mask)
            - input_ids: [L] tensor of token ids
            - attention_mask: [L] tensor of attention mask
            - target_mask: [L] tensor where 1 indicates target word tokens, 0 otherwise
        """
        # Tokenize the sentence
        encoding = self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            add_special_tokens=True
        )
        
        input_ids = encoding['input_ids'].squeeze(0)  # [L]
        attention_mask = encoding['attention_mask'].squeeze(0)  # [L]
        
        # Find target word positions
        target_mask = self._find_target_positions(sentence, target_word, input_ids, case_sensitive)
        
        return input_ids, attention_mask, target_mask
    
    def extract_target_mask_batch(self, sentences, target_words, 
                                case_sensitive: bool = False, max_length: Optional[int] = None) -> dict:
        """
        Extract masks for a batch of sentences and target words.
        
        Args:
            sentences: List of input sentences
            target_words: List of target words (one for each sentence)
            case_sensitive: Whether to match case sensitively
            max_length: Maximum sequence length for padding
            
        Returns:
            Dictionary with keys: 'input_ids', 'attention_mask', 'target_mask'
            Each value is a tensor of shape [B, L]
        """
        batch_input_ids = []
        batch_attention_mask = []
        batch_target_mask = []
        
        # Process each sentence-target pair
        for sentence, target_word in zip(sentences, target_words):
            input_ids, attention_mask, target_mask = self.extract_target_mask(
                sentence, target_word, case_sensitive
            )
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_target_mask.append(target_mask)
        
        # Pad sequences to same length
        max_len = max_length or max(len(ids) for ids in batch_input_ids)
        
        # Pad and stack
        padded_input_ids = []
        padded_attention_mask = []
        padded_target_mask = []
        
        for input_ids, attention_mask, target_mask in zip(batch_input_ids, batch_attention_mask, batch_target_mask):
            # Pad sequences
            pad_length = max_len - len(input_ids)
            
            padded_input_ids.append(torch.cat([
                input_ids, 
                torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)
            ]))
            
            padded_attention_mask.append(torch.cat([
                attention_mask,
                torch.zeros(pad_length, dtype=torch.long)
            ]))
            
            padded_target_mask.append(torch.cat([
                target_mask,
                torch.zeros(pad_length, dtype=torch.long)
            ]))
        
        return {
            'input_ids': torch.stack(padded_input_ids),
            'attention_mask': torch.stack(padded_attention_mask),
            'target_mask': torch.stack(padded_target_mask)
        }
    
    def _find_target_positions(self, sentence: str, target_word: str, input_ids: torch.Tensor, case_sensitive: bool = False) -> torch.Tensor:
        """
        Find positions of target word tokens in the tokenized sequence.
        
        Args:
            sentence: Original sentence
            target_word: Target word to find
            input_ids: Tokenized input ids
            case_sensitive: Whether to match case sensitively
            
        Returns:
            target_mask: [L] tensor where 1 indicates target word tokens
        """
        target_mask = torch.zeros_like(input_ids, dtype=torch.long)
        
        # Convert input_ids back to tokens to find word boundaries
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        # Handle case sensitivity
        search_word = target_word if case_sensitive else target_word.lower()
        search_tokens = [token if case_sensitive else token.lower() for token in tokens]
        
        # Tokenize target word separately to handle subword tokenization
        target_encoding = self.tokenizer(
            target_word,
            add_special_tokens=False,
            return_tensors="pt"
        )
        target_tokens = self.tokenizer.convert_ids_to_tokens(target_encoding['input_ids'].squeeze(0))
        
        if not case_sensitive:
            target_tokens = [token.lower() for token in target_tokens]
        
        # Find target word token sequence in the sentence tokens
        target_positions = self._find_subword_sequence(search_tokens, target_tokens)
        
        # Set mask for target positions
        for start_pos, end_pos in target_positions:
            target_mask[start_pos:end_pos] = 1
        
        return target_mask
    
    def _find_subword_sequence(self, tokens, target_tokens):
        """
        Find all occurrences of target token sequence in tokens list.
        
        Args:
            tokens: List of tokens from sentence
            target_tokens: List of target word tokens
            
        Returns:
            List of (start_pos, end_pos) tuples indicating target word positions
        """
        positions = []
        target_len = len(target_tokens)
        
        if target_len == 0:
            return positions
        
        for i in range(len(tokens) - target_len + 1):
            # Check if we have a match at position i
            match = True
            for j in range(target_len):
                # Handle special tokens and subword prefixes (like ## in BERT, @@in PhoBERT)
                token = tokens[i + j]
                target_token = target_tokens[j]
                
                # Remove subword prefixes for comparison
                clean_token = self._clean_subword_token(token)
                clean_target = self._clean_subword_token(target_token)
                
                if clean_token != clean_target:
                    match = False
                    break
            
            if match:
                positions.append((i, i + target_len))
        
        return positions
    
    def _clean_subword_token(self, token: str) -> str:
        """
        Clean subword token by removing special prefixes.
        
        Args:
            token: Token to clean
            
        Returns:
            Cleaned token
        """
        # Handle PhoBERT subword tokens (@@)
        if token.startswith('@@'):
            return token[2:]
        # Handle BERT-style subword tokens (##)
        elif token.startswith('##'):
            return token[2:]
        # Handle other special tokens
        elif token.startswith('▁'):  # SentencePiece
            return token[1:]
        
        return token
    
    def visualize_mask(self, sentence: str, target_word: str, case_sensitive: bool = False) -> str:
        """
        Visualize the target mask for debugging purposes.
        
        Args:
            sentence: Input sentence
            target_word: Target word
            case_sensitive: Whether to match case sensitively
            
        Returns:
            String showing tokens and their mask status
        """
        input_ids, attention_mask, target_mask = self.extract_target_mask(
            sentence, target_word, case_sensitive
        )
        
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        result = []
        result.append(f"Sentence: {sentence}")
        result.append(f"Target word: {target_word}")
        result.append("Token visualization:")
        
        for i, (token, mask_val) in enumerate(zip(tokens, target_mask)):
            marker = ">>>" if mask_val == 1 else "   "
            result.append(f"{marker} {i:2d}: {token}")
        
        return "\n".join(result)


# Example usage and testing
if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    # Initialize tokenizer (PhoBERT)
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    extractor = TargetWordMaskExtractor(tokenizer)
    
    # Test single sentence
    sentence = "Tôi đang học machine learning và deep learning."
    target_word = "learning"
    
    print("=== Single Sentence Test ===")
    print(extractor.visualize_mask(sentence, target_word))
    
    # Test batch processing
    sentences = [
        "Tôi đang học machine learning và deep learning.",
        "Ngôn ngữ Python rất phù hợp cho machine learning.",
        "Deep learning là một phần của machine learning."
    ]
    target_words = ["learning", "Python", "learning"]
    
    print("\n=== Batch Processing Test ===")
    batch_result = extractor.extract_target_mask_batch(sentences, target_words)
    
    print(f"Batch input_ids shape: {batch_result['input_ids'].shape}")
    print(f"Batch attention_mask shape: {batch_result['attention_mask'].shape}")
    print(f"Batch target_mask shape: {batch_result['target_mask'].shape}")
    
    # Show first example from batch
    print(f"\nFirst example target_mask: {batch_result['target_mask'][0]}")