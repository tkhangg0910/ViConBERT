import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import torch

class SpanExtractor:
    """Extracts start and end indices for target phrases in Vietnamese text"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_span_indices(self, text, target_phrase):
        """
        Finds start and end token indices for target phrase
        Handles Vietnamese subword tokenization
        
        Args:
            text: Full input text
            target_phrase: Target phrase to locate
            
        Returns:
            Tuple (start_idx, end_idx) if found, else None
        """
        # Find character-level position
        start_char = text.find(target_phrase)
        if start_char == -1:
            return None
        
        end_char = start_char + len(target_phrase)
        
        # Tokenize with offset mapping
        encoding = self.tokenizer(text, return_offsets_mapping=True, truncation=True, max_length=512)
        offsets = encoding["offset_mapping"]
        
        # Find tokens within target span
        start_idx, end_idx = None, None
        for i, (char_start, char_end) in enumerate(offsets):
            if char_end <= start_char:
                continue
            if char_start >= end_char:
                break
            if char_start >= start_char and char_end <= end_char:
                if start_idx is None:  # First token in span
                    start_idx = i
                end_idx = i  # Update last token in span
        
        return (start_idx, end_idx) if start_idx is not None else None

class FusionBlock(nn.Module):
    """Neural block to combine CLS and span representations"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim: Dimension of concatenated features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) 
        self.activation = nn.GELU() 
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, std=1e-6)
    
    def forward(self, x):
        """Forward pass through fusion block"""
        x = self.fc1(x)
        x = self.norm(x)
        x = self.activation(x)
        return self.fc2(x)

class SynoViSenseEmbedding(nn.Module):
    """Vietnamese Contextual Embedding Model with Span Representation"""
    def __init__(self, model_name, fb_hdim=512):
        """
        Args:
            model_name: PhoBERT model name
            fb_hdim: Fusion block hidden dimension
        """
        super().__init__()
        # Initialize base model and tokenizer
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Span processing components
        self.extractor = SpanExtractor(self.tokenizer)
        
        # Fusion block for combining CLS and span representations
        self.fusion = FusionBlock(
            input_dim=4 * self.hidden_size,  # CLS (h) + span (3h)
            hidden_dim=fb_hdim,
            output_dim=self.hidden_size      # Final embedding size
        )
        
    def forward(self, input_ids, attention_mask, target_phrases, texts):
        """
        Forward pass to generate contextual embeddings
        
        Args:
            input_ids: Token IDs from tokenizer
            attention_mask: Attention mask for padding
            target_phrases: List of target phrases
            texts: Original input texts
            
        Returns:
            Contextual embeddings [batch_size, hidden_size]
        """
        # Base model forward pass
        outputs = self.base_model(
            input_ids, 
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        
        # Get CLS embedding (global context)
        cls_embed = hidden_states[:, 0, :]
        
        # Get span representation (local context)
        span_rep = self._get_span_representation(hidden_states, texts, target_phrases)
        
        # Combine global and local features
        combined = torch.cat([cls_embed, span_rep], dim=-1)
        
        # Fuse features through neural block
        return self.fusion(combined)
    
    def _get_span_representation(self, hidden_states, texts, target_phrases):
        """
        Generate span representations for a batch
        
        Args:
            hidden_states: Hidden states from base model [batch_size, seq_len, hidden_size]
            texts: Original texts [batch_size]
            target_phrases: Target phrases [batch_size]
            
        Returns:
            Span representations [batch_size, 3 * hidden_size]
        """
        batch_size = hidden_states.size(0)
        span_embeddings = []
        
        for i in range(batch_size):
            # Get span indices for target phrase
            span_indices = self.extractor.get_span_indices(texts[i], target_phrases[i])
            
            if span_indices:
                start_idx, end_idx = span_indices
                # Handle single token case
                if start_idx == end_idx:
                    start_embed = hidden_states[i, start_idx, :].unsqueeze(0)
                    end_embed = start_embed
                    diff_embed = torch.zeros_like(start_embed)
                    span_emb = torch.cat([start_embed, end_embed, diff_embed], dim=-1)
                else:
                    span_emb = self._get_single_span_embedding(
                        hidden_states[i].unsqueeze(0), 
                        start_idx, 
                        end_idx
                    )
            else:
                # Fallback: use CLS embedding if span not found
                span_emb = hidden_states[i, 0, :].unsqueeze(0)
                # Create zero-padded span representation
                zeros = torch.zeros_like(span_emb)
                span_emb = torch.cat([span_emb, zeros, zeros], dim=-1)
            
            span_embeddings.append(span_emb)
        
        return torch.cat(span_embeddings, dim=0)
    
    def _get_single_span_embedding(self, hidden_states, start_idx, end_idx):
        """
        Create span representation for a single instance
        
        Args:
            hidden_states: Hidden states [1, seq_len, hidden_size]
            start_idx: Start token index
            end_idx: End token index
            
        Returns:
            Span representation [1, 3 * hidden_size]
        """
        start_embed = hidden_states[:, start_idx, :]
        end_embed = hidden_states[:, end_idx, :]
        diff_embed = end_embed - start_embed
        return torch.cat([start_embed, end_embed, diff_embed], dim=-1)

    def tokenize_with_target(self, texts, target_phrases):
        """
        Tokenize batch with span position detection
        
        Args:
            texts: List of input texts
            target_phrases: List of target phrases
            
        Returns:
            Dictionary with input_ids, attention_mask, and span positions
        """
        # Tokenize batch
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            return_offsets_mapping=True
        )
        
        # Get span positions
        span_starts, span_ends = [], []
        for text, target_phrase in zip(texts, target_phrases):
            indices = self.extractor.get_span_indices(text, target_phrase)
            if indices:
                span_starts.append(indices[0])
                span_ends.append(indices[1])
            else:
                # Fallback to CLS position
                span_starts.append(0)
                span_ends.append(0)
        
        inputs["span_starts"] = torch.tensor(span_starts)
        inputs["span_ends"] = torch.tensor(span_ends)
        return inputs