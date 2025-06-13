import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Optional, Tuple, Union
import logging

class SpanExtractor:
    """Extracts start and end indices for target phrases in Vietnamese text"""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(__name__)
    
    def get_span_indices(self, text: str, target_phrase: str) -> Optional[Tuple[int, int]]:
        """
        Finds start and end token indices for target phrase
        Handles Vietnamese subword tokenization with improved accuracy
        
        Args:
            text: Full input text
            target_phrase: Target phrase to locate
            
        Returns:
            Tuple (start_idx, end_idx) if found, else None
        """
        # Normalize whitespace for better matching
        text = ' '.join(text.split())
        target_phrase = ' '.join(target_phrase.split())
        
        # Find character-level position
        start_char = text.find(target_phrase)
        if start_char == -1:
            self.logger.warning(f"Target phrase '{target_phrase}' not found in text")
            return None
        
        end_char = start_char + len(target_phrase)
        
        # Tokenize with offset mapping
        encoding = self.tokenizer(
            text, 
            return_offsets_mapping=True, 
            truncation=True, 
            max_length=512,
            add_special_tokens=True
        )
        offsets = encoding["offset_mapping"]
        
        # Find tokens within target span
        start_idx, end_idx = None, None
        
        for i, (char_start, char_end) in enumerate(offsets):
            # Skip special tokens with (0, 0) offsets
            if char_start == 0 and char_end == 0 and i > 0:
                continue
                
            # Check if token overlaps with target span
            if char_end <= start_char:
                continue
            if char_start >= end_char:
                break
                
            # Token is within or overlaps target span
            if start_idx is None and char_start < end_char:
                start_idx = i
            if char_start < end_char:
                end_idx = i
        
        if start_idx is not None and end_idx is not None:
            return (start_idx, end_idx)
        
        self.logger.warning(f"Could not find token indices for '{target_phrase}'")
        return None

class FusionBlock(nn.Module):
    """Enhanced neural block to combine CLS and span representations"""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of concatenated features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/He initialization"""
        for layer in [self.fc1, self.fc2, self.fc_out]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through enhanced fusion block"""
        # First layer with residual connection
        residual = x
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        x = x + residual
        
        # Second layer with residual connection
        residual = x
        x = self.fc2(x)
        x = self.norm2(x)
        x = self.dropout2(x)
        x = self.activation(x)
        x = x + residual
        # Output layer
        x = self.fc_out(x)
        
        return x

class SynoViSenseEmbedding(nn.Module):
    """
    Enhanced Vietnamese Contextual Embedding Model with Flexible Span Representation
    Optimized for training/inference performance with precomputed span indices
    """
    
    def __init__(self, 
                 model_name: str, 
                 fb_hdim: int = 512, 
                 span_method: str = "span",
                 dropout: float = 0.1,
                 freeze_base: bool = False):
        """
        Args:
            model_name: PhoBERT model name (e.g., "vinai/phobert-base")
            fb_hdim: Fusion block hidden dimension
            span_method: Span representation method ("span", "mean")
            dropout: Dropout rate for fusion block
            freeze_base: Whether to freeze base model parameters
        """
        super().__init__()
        self.span_method = span_method
        self.logger = logging.getLogger(__name__)
        
        # Initialize base model and tokenizer
        self.base_model = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.hidden_size = self.base_model.config.hidden_size
        
        # Freeze base model if specified
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Span processing components
        self.extractor = SpanExtractor(self.tokenizer)
        
        # Adjust input dim based on span method
        if span_method == "span":
            input_dim = 4 * self.hidden_size  # CLS + start + end + diff
        elif span_method in ["mean"]:
            input_dim = 2 * self.hidden_size  # CLS + pooled span
        else:
            raise ValueError(f"Unsupported span_method: {span_method}")
        
        # Enhanced fusion block
        self.fusion = FusionBlock(
            input_dim=input_dim,
            hidden_dim=fb_hdim,
            output_dim=self.hidden_size,
            dropout=dropout
        )
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                target_phrases: Optional[List[str]] = None,
                texts: Optional[List[str]] = None,
                span_indices: Optional[List[Optional[Tuple[int, int]]]] = None) -> torch.Tensor:
        """
        Forward pass to generate contextual embeddings
        
        Args:
            input_ids: Token IDs from tokenizer [batch_size, seq_len]
            attention_mask: Attention mask for padding [batch_size, seq_len]
            target_phrases: List of target phrases (for inference)
            texts: Original input texts (for inference)
            span_indices: Precomputed span indices (for training)
            
        Returns:
            Contextual embeddings [batch_size, hidden_size]
        """
        # Base model forward pass
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        hidden_states = outputs.last_hidden_state
        
        # Compute span indices if not provided (inference mode)
        if span_indices is None:
            if target_phrases is None or texts is None:
                raise ValueError("Must provide either span_indices or (target_phrases, texts)")
            span_indices = self._compute_span_indices(texts, target_phrases)
        
        # Get CLS embedding (global context)
        cls_embed = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        
        # Get span representation (local context)
        span_rep = self._get_span_representation(hidden_states, span_indices)
        
        # Combine global and local features
        combined = torch.cat([cls_embed, span_rep], dim=-1)
        
        # Fuse features through neural block
        return self.fusion(combined)
    
    def _get_span_representation(self, 
                               hidden_states: torch.Tensor, 
                               span_indices: List[Optional[Tuple[int, int]]]) -> torch.Tensor:
        """
        Generate span representations for a batch (vectorized)
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_size]
            span_indices: List of (start_idx, end_idx) tuples
            
        Returns:
            Span representations [batch_size, span_dim]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        if self.span_method == "span":
            return self._get_span_based_embedding(hidden_states, span_indices)
        elif self.span_method == "mean":
            return self._get_mean_pooled_embedding(hidden_states, span_indices)
        else:
            # Fallback to CLS for all samples
            return hidden_states[:, 0, :]
    
    def _get_span_based_embedding(self, hidden_states: torch.Tensor, 
                                      span_indices: List[Optional[Tuple[int, int]]]) -> torch.Tensor:
        """span representation: [start, end, end-start]"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Prepare batch indices
        batch_idx = torch.arange(batch_size, device=device)
        start_indices = torch.zeros(batch_size, dtype=torch.long, device=device)  # Default to CLS
        end_indices = torch.zeros(batch_size, dtype=torch.long, device=device)    # Default to CLS
        valid_spans = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Fill valid span indices
        for i, indices in enumerate(span_indices):
            if indices is not None:
                start_idx, end_idx = indices
                # Ensure indices are within bounds
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx, min(end_idx, seq_len - 1))
                
                start_indices[i] = start_idx
                end_indices[i] = end_idx
                valid_spans[i] = True
        
        start_embeds = hidden_states[batch_idx, start_indices]  # [batch_size, hidden_size]
        end_embeds = hidden_states[batch_idx, end_indices]      # [batch_size, hidden_size]
        diff_embeds = end_embeds - start_embeds                 # [batch_size, hidden_size]
        
        # For invalid spans, use CLS embedding for all components
        cls_embeds = hidden_states[:, 0, :]  # [batch_size, hidden_size]
        zeros = torch.zeros_like(cls_embeds)
        
        # Apply mask for invalid spans
        start_embeds = torch.where(valid_spans.unsqueeze(-1), start_embeds, cls_embeds)
        end_embeds = torch.where(valid_spans.unsqueeze(-1), end_embeds, zeros)
        diff_embeds = torch.where(valid_spans.unsqueeze(-1), diff_embeds, zeros)
        
        return torch.cat([start_embeds, end_embeds, diff_embeds], dim=-1)
    
    
    def _get_mean_pooled_embedding(self, hidden_states: torch.Tensor,
                                       span_indices: List[Optional[Tuple[int, int]]]) -> torch.Tensor:
        """mean pooling"""
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device
        
        # Create mask for span tokens
        span_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        for i, indices in enumerate(span_indices):
            if indices is not None:
                start_idx, end_idx = indices
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx, min(end_idx, seq_len - 1))
                span_mask[i, start_idx:end_idx+1] = True
            else:
                # Use CLS token for invalid spans
                span_mask[i, 0] = True
        
        # Apply mask and compute mean
        masked_hidden = hidden_states * span_mask.unsqueeze(-1).float()
        span_lengths = span_mask.sum(dim=-1, keepdim=True).float()  # [batch_size, 1]
        span_lengths = torch.clamp(span_lengths, min=1.0)  # Avoid division by zero
        
        span_embeddings = masked_hidden.sum(dim=1) / span_lengths  # [batch_size, hidden_size]
        
        return span_embeddings
    
    def _compute_span_indices(self, texts: List[str], target_phrases: List[str]) -> List[Optional[Tuple[int, int]]]:
        """Compute span indices for batch (used during inference)"""
        return [
            self.extractor.get_span_indices(text, phrase)
            for text, phrase in zip(texts, target_phrases)
        ]
    
    def tokenize_with_target(self, texts: List[str], target_phrases: List[str]) -> dict:
        """
        Tokenize texts and compute span indices (optimized for training)
        
        Args:
            texts: List of input texts
            target_phrases: List of target phrases
            
        Returns:
            Dictionary with tokenized inputs and span indices
        """
        # Tokenize texts
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        # Precompute span indices
        inputs["span_indices"] = self._compute_span_indices(texts, target_phrases)
        inputs["texts"] = texts
        inputs["target_phrases"] = target_phrases
        
        return inputs
    
    def encode(self, texts: List[str], target_phrases: List[str]) -> torch.Tensor:
        """
        Convenience method for encoding texts with target phrases
        
        Args:
            texts: List of input texts
            target_phrases: List of target phrases
            
        Returns:
            Encoded embeddings [batch_size, hidden_size]
        """
        inputs = self.tokenize_with_target(texts, target_phrases)
        
        with torch.no_grad():
            embeddings = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                span_indices=inputs["span_indices"]
            )
        
        return embeddings
