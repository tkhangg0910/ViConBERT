import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerFast
from typing import List, Optional, Tuple, Dict
from utils.span_extractor import SpanExtractor

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
        x = self.fc1(x)
        x = self.norm1(x)
        x = self.dropout1(x)
        x = self.activation(x)
        
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

class AttentivePooling(nn.Module):
    def __init__(self, hidden_size, attn_hidden=128):
        """
        Attentive Pooling Layer
        Args:
            hidden_size: size of embedding (d)
            attn_hidden: hidden size of attention (tùy chọn)
        """
        super().__init__()
        self.W = nn.Linear(hidden_size, attn_hidden)
        self.u = nn.Linear(attn_hidden, 1, bias=False)
        
    def forward(self, embeddings, mask=None):
        """
        Args:
            embeddings: Tensor shape [batch_size, seq_len, hidden_size]
            mask: Tensor shape [batch_size, seq_len] (1 = valid token, 0 = pad)
        Returns:
            pooled: Tensor shape [batch_size, hidden_size]
        """
        # Tính attention scores
        e = torch.tanh(self.W(embeddings))  # [batch, seq_len, attn_hidden]
        scores = self.u(e).squeeze(-1)      # [batch, seq_len]
        
        # Xử lý mask
        if mask is not None:
            scores = scores.masked_fill(~mask, float('-inf'))
        
        # Tính trọng số attention
        alpha = F.softmax(scores, dim=-1)   # [batch, seq_len]
        
        # Tổng hợp có trọng số
        pooled = torch.sum(embeddings * alpha.unsqueeze(-1), dim=1)  # [batch, hidden_size]
        return pooled
    
class LayerwiseCLSPooling(nn.Module):
    def __init__(self, hidden_size, layer_attn_hidden=128):
        """
        Layerwise CLS Pooling
        Args:
            hidden_size: embedding size (d)
            layer_attn_hidden:  attention's hidden size
        """
        super().__init__()
        self.U = nn.Linear(hidden_size, layer_attn_hidden)
        self.v = nn.Linear(layer_attn_hidden, 1, bias=False)
        
    def forward(self, all_layer_cls_embeddings):
        """
        Args:
            all_layer_cls_embeddings: 
                List of tensors [CLS] from each layer, 
                each tensor shape [batch_size, hidden_size]
                or tensor [batch_size, num_layers, hidden_size]
        Returns:
            pooled: Tensor shape [batch_size, hidden_size]
        """
        if isinstance(all_layer_cls_embeddings, list):
            all_layer_cls = torch.stack(all_layer_cls_embeddings, dim=1)  # [batch, L, d]
        else:
            all_layer_cls = all_layer_cls_embeddings
        
        # Tính attention scores cho các layer
        e = torch.tanh(self.U(all_layer_cls))  # [batch, L, layer_attn_hidden]
        scores = self.v(e).squeeze(-1)         # [batch, L]
        
        # Tính trọng số layer attention
        beta = F.softmax(scores, dim=-1)       # [batch, L]
        
        # Tổng hợp có trọng số
        pooled = torch.sum(all_layer_cls * beta.unsqueeze(-1), dim=1)  # [batch, d]
        return pooled

class FixedRobertaModel(nn.Module):
    """Wrapper to completely disable token_type_ids in RoBERTa"""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config
        
    def forward(self, input_ids, attention_mask, **kwargs):
        # Always override token_type_ids to None
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            **kwargs
        )
class SynoViSenseEmbedding(nn.Module):
    """
    Enhanced Vietnamese Contextual Embedding Model
    with multiple span representation options
    """
    
    def __init__(self, 
                 tokenizer,
                 model_name: str = "vinai/phobert-base",
                 cache_dir: str ="embeddings/base_models",
                 fusion_hidden_dim: int = 512,
                 span_method: str = "attentive",
                 cls_method: str = "layerwise",
                 dropout: float = 0.1,
                 freeze_base: bool = False,
                 layerwise_attn_dim: int = 128):
        """
        Args:
            model_name: Pre-trained model name
            fusion_hidden_dim: Fusion block hidden dimension
            span_method: Span representation method 
                         ("diff_sum", "mean", "attentive")
            cls_method: cls representation method 
                         ("layerwise", "last")
            dropout: Dropout rate
            freeze_base: Freeze base model parameters
            layerwise_attn_dim: Attention dim for layerwise pooling
        """
        super().__init__()
        self.span_method = span_method
        self.cls_method = cls_method
        self.logger = logging.getLogger(__name__)
        
        # Initialize base model
        base = AutoModel.from_pretrained(
            model_name,
            output_hidden_states=(cls_method == "layerwise")
            ,cache_dir=cache_dir
        )
        self.base_model = FixedRobertaModel(base)
        if hasattr(self.base_model, "token_type_ids"):
            del self.base_model.token_type_ids
        self.base_model.register_buffer("token_type_ids", None)

        self.tokenizer = tokenizer
        self.hidden_size = self.base_model.config.hidden_size
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        # Span processing components
        self.extractor = SpanExtractor(self.tokenizer)
        
        # Initialize pooling modules
        if span_method == "attentive":
            self.attentive_pool = AttentivePooling(self.hidden_size)
        
        elif cls_method == "layerwise":
            self.layerwise_pool = LayerwiseCLSPooling(self.hidden_size, layerwise_attn_dim)
        
        # Determine fusion input dimension
        if span_method == "diff_sum":
            input_dim = 4 * self.hidden_size  # CLS + (start, end, diff)
        else:
            input_dim = 2 * self.hidden_size  # CLS + span representation
        
        # Fusion block
        self.fusion = FusionBlock(
            input_dim=input_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.base_model.resize_token_embeddings(len(tokenizer))

        
        # Layer normalization for span representations
        self.span_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor, 
                target_phrases: Optional[List[str]] = None,
                texts: Optional[List[str]] = None,
                span_indices: Optional[List[Optional[Tuple[int, int]]]] = None) -> torch.Tensor:
        """
        Forward pass with flexible span representation
        """
        # Base model forward
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=None,
            output_hidden_states=(self.cls_method == "layerwise")
        )
        
        last_hidden_state = outputs.last_hidden_state
        
        # Get CLS representation
        if self.cls_method == "layerwise":
            all_hidden_states = outputs.hidden_states
            cls_embed = self.layerwise_pool(
                [hidden[:, 0] for hidden in all_hidden_states[1:]]
                )
        else:
            cls_embed = last_hidden_state[:, 0, :]
        
        
        # Compute span indices if needed
        if span_indices is None:
            if target_phrases is None or texts is None:
                raise ValueError("Must provide span indices or (texts, target_phrases)")
            span_indices = self._compute_span_indices(texts, target_phrases)
        
        # Get span representation
        span_rep = self._get_span_representation(
            last_hidden_state, 
            span_indices,
        )
        
        # Combine and fuse representations
        combined = torch.cat([cls_embed, span_rep], dim=-1)
        return self.fusion(combined)
    
    def _get_span_representation(self, 
                               hidden_states: torch.Tensor, 
                               span_indices: List[Optional[Tuple[int, int]]]
                               ) -> torch.Tensor:
        """
        Flexible span representation with multiple methods
        """
        if self.span_method == "diff_sum":
            return self._get_diff_sum_based_embedding(hidden_states, span_indices)
        elif self.span_method == "mean":
            return self._get_mean_pooled_embedding(hidden_states, span_indices)
        elif self.span_method == "attentive":
            return self._get_attentive_pooled_embedding(hidden_states, span_indices)
        else:
            raise ValueError(f"Unsupported span method: {self.span_method}")
    
    def _get_diff_sum_based_embedding(self, 
                                hidden_states: torch.Tensor, 
                                span_indices: List[Optional[Tuple[int, int]]]) -> torch.Tensor:
        """Start/end/difference representation"""
        batch_size, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device
        
        # Initialize with CLS as fallback
        start_embeds = hidden_states[:, 0, :].clone()
        end_embeds = torch.zeros_like(start_embeds)
        diff_embeds = torch.zeros_like(start_embeds)
        
        # Process valid spans
        for i, indices in enumerate(span_indices):
            if indices is not None:
                start_idx, end_idx = indices
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx, min(end_idx, seq_len - 1))
                
                start_embeds[i] = hidden_states[i, start_idx]
                end_embeds[i] = hidden_states[i, end_idx]
                diff_embeds[i] = end_embeds[i] - start_embeds[i]
        
        return torch.cat([start_embeds, end_embeds, diff_embeds], dim=-1)
    
    def _get_mean_pooled_embedding(self, 
                                 hidden_states: torch.Tensor,
                                 span_indices: List[Optional[Tuple[int, int]]]) -> torch.Tensor:
        """Mean pooling over span"""
        batch_size, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device
        span_embeddings = torch.zeros(batch_size, hidden_size, device=device)
        
        for i, indices in enumerate(span_indices):
            if indices is not None:
                start_idx, end_idx = indices
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx, min(end_idx, seq_len - 1))
                
                span_tokens = hidden_states[i, start_idx:end_idx+1]
                span_embeddings[i] = span_tokens.mean(dim=0)
            else:
                span_embeddings[i] = hidden_states[i, 0]  # CLS fallback
        
        return self.span_norm(span_embeddings)
    
    def _get_attentive_pooled_embedding(self, 
                                      hidden_states: torch.Tensor,
                                      span_indices: List[Optional[Tuple[int, int]]]) -> torch.Tensor:
        """Attentive pooling over span"""
        batch_size, seq_len, hidden_size = hidden_states.size()
        device = hidden_states.device
        span_embeddings = torch.zeros(batch_size, hidden_size, device=device)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        
        # Create span masks
        for i, indices in enumerate(span_indices):
            if indices is not None:
                start_idx, end_idx = indices
                start_idx = max(0, min(start_idx, seq_len - 1))
                end_idx = max(start_idx, min(end_idx, seq_len - 1))
                mask[i, start_idx:end_idx+1] = True
            else:
                mask[i, 0] = True  # CLS fallback
        
        # Apply attentive pooling
        span_embeddings = self.attentive_pool(hidden_states, mask)
        return self.span_norm(span_embeddings)
        
    
    def _compute_span_indices(self, 
                            texts: List[str], 
                            target_phrases: List[str]) -> List[Optional[Tuple[int, int]]]:
        """Batch span index computation"""
        return [
            self.extractor.get_span_indices(text, phrase)
            for text, phrase in zip(texts, target_phrases)
        ]
    
    def tokenize_with_target(self, 
                           texts: List[str], 
                           target_phrases: List[str]) -> Dict[str, torch.Tensor]:
        """
        Optimized tokenization with span index precomputation
        """
        inputs = self.tokenizer(
            texts, 
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Precompute span indices
        inputs["span_indices"] = self._compute_span_indices(texts, target_phrases)
        return inputs
    
    def encode(self, 
             texts: List[str], 
             target_phrases: List[str]) -> torch.Tensor:
        """
        High-level encoding API
        """
        inputs = self.tokenize_with_target(texts, target_phrases)
        
        with torch.no_grad():
            embeddings = self.forward(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                span_indices=inputs["span_indices"]
            )
        
        return embeddings
