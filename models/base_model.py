import logging
import torch
import torch.nn as nn
from transformers import AutoModel
from typing import List, Tuple
from typing import Callable

class MLPBlock(nn.Module):
    """Enhanced neural block to combine context and word representations"""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 num_layers: int = 2, 
                 dropout: float = 0.1,
                 activation: Callable = nn.GELU,
                 use_residual: bool = True,
                 final_activation = None):
        """
        Args:
            input_dim: Dimension of concatenated features
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.use_residual = use_residual
        self.activation_fn = activation()

        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.final_activation=None
        if final_activation:
            self.final_activation = final_activation
        self.hidden_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout))
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.normal_(self.input_layer.bias, std=1e-6)
        
        for layer in self.hidden_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.normal_(layer.bias, std=1e-6)
            
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.normal_(self.output_layer.bias, std=1e-6)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through enhanced fusion block"""
        x = self.input_layer(x)
        for layer, norm, dropout in zip(self.hidden_layers, self.norms, self.dropouts):
            residual = x
            x = layer(x)
            x = norm(x)
            x = dropout(x)
            x = self.activation_fn(x)
            if self.use_residual:
                x = x + residual
        if self.final_activation:
            x = self.final_activation(x)
        x = self.output_layer(x)

        
        return x

class SynoViSenseEmbeddingV1(nn.Module):
    """
    Enhanced Vietnamese Contextual Embedding Model
    with multiple span representation options
    """
    
    def __init__(self, 
                 tokenizer,
                 model_name: str = "vinai/phobert-base",
                 cache_dir: str ="embeddings/base_models",
                 fusion_hidden_dim: int = 512,
                 wp_num_layers:int=1,
                 cp_num_layers:int=1,
                 dropout: float = 0.1,
                 freeze_base: bool = False,
                 fusion_num_layers:int=1
                 ):
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
        self.logger = logging.getLogger(__name__)
        
        # Initialize base model
        self.base_model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        self.base_model.resize_token_embeddings(len(tokenizer))

        
        self.tokenizer = tokenizer
        self.hidden_size = self.base_model.config.hidden_size
        
        # Freeze base model if requested
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        
        self.context_projection = MLPBlock(
            input_dim=self.hidden_size,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=cp_num_layers
        )
        
        self.word_projection = MLPBlock(
            input_dim=self.hidden_size,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=wp_num_layers
        )
        
        self.fusion_gate = MLPBlock(
            input_dim=self.hidden_size*2,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=fusion_num_layers,
            final_activation=nn.Sigmoid()

        )
        
        # Layer normalization for span representations
        self.span_norm = nn.LayerNorm(self.hidden_size)
        
    def forward(self, 
                word_input_ids: torch.Tensor,
                word_attention_mask: torch.Tensor,
                context_input_ids: torch.Tensor,
                context_attention_mask: torch.Tensor,
                target_spans =None):
        """
        Forward pass with flexible span representation
        """
        word_emb = self._encode_word(
            word_input_ids,
            word_attention_mask
        )
        context_emb = self._encode_context(
            context_input_ids,
            context_attention_mask
        )
        combined = torch.cat([word_emb, context_emb], dim=-1)
        gate = self.fusion_gate(combined)
        fused_embed = gate * word_emb + (1 - gate) * context_emb

        return fused_embed
        
    def _encode_word(self, word_input_ids, word_attention_mask):
        outputs = self.base_model(
            input_ids=word_input_ids,
            attention_mask=word_attention_mask
        )
        word_rep = outputs.last_hidden_state[:, 0] 
        return self.word_projection(word_rep)

    def _encode_context(self, context_input_ids, context_attention_mask):
        outputs = self.base_model(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_rep = outputs.last_hidden_state[:, 0]  
        return self.context_projection(context_rep)

class ContextAttention(nn.Module):
    def __init__(self, hidden_size, num_heads=4, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, context, attention_mask):
        # query: [batch_size, hidden_dim]
        # context: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len]
        
        query = query.unsqueeze(1)
        attn_output, _ = self.attention(
            query=query,
            key=context,
            value=context,
            key_padding_mask=~attention_mask.bool()
        )
        attn_output = self.dropout(attn_output.squeeze(1))
        return self.norm(attn_output + query.squeeze(1))



class SynoViSenseEmbeddingV2(nn.Module):
    def __init__(self, 
        tokenizer,
        model_name: str = "vinai/phobert-base",
        cache_dir: str ="embeddings/base_models",
        fusion_hidden_dim: int = 512,
        wp_num_layers:int=1,
        cp_num_layers:int=1,
        dropout: float = 0.1,
        freeze_base: bool = False,
        fusion_num_layers:int=1,
        context_window_size:int = 3,
        use_context_attention: bool = True,
        context_num_heads: int = 4,
        ):
        super().__init__()
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
        self.context_window_size = context_window_size
        self.tokenizer = tokenizer
        self.base_model = AutoModel.from_pretrained(model_name, 
                                              cache_dir=cache_dir)
        self.hidden_size = self.base_model.config.hidden_size
        self.use_context_attention = use_context_attention
        self.base_model.resize_token_embeddings(len(tokenizer))
        
        self.fusion_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=context_num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.fusion_norm = nn.LayerNorm(self.hidden_size)
        self.fusion_dropout = nn.Dropout(dropout)

        
        self.context_proj = MLPBlock(
            input_dim=self.hidden_size,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=cp_num_layers
        )
        self.word_projection = MLPBlock(
            input_dim=self.hidden_size,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=wp_num_layers
        )
        self.output_proj = MLPBlock(
            input_dim=self.hidden_size,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=fusion_num_layers
        )

        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
                
        if self.use_context_attention:
            self.context_attention = ContextAttention(
                hidden_size=self.hidden_size,
                num_heads=context_num_heads,
                dropout=dropout
            )

    def _encode_word(self, word_input_ids, word_attention_mask):
        outputs = self.base_model(
            input_ids=word_input_ids,
            attention_mask=word_attention_mask
        )
        word_rep = outputs.last_hidden_state[:, 0] 
        return self.word_projection(word_rep)
    
    def _encode_context(self, context_input_ids, context_attention_mask, target_spans):
        outputs = self.base_model(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask
        )
        context_embeddings = outputs.last_hidden_state
        
        # Get target span representation
        batch_size = context_embeddings.size(0)
        span_reps = []
        
        for i in range(batch_size):
            start_pos = max(0, min(target_spans[i, 0], context_embeddings.size(1)-1))
            end_pos = max(0, min(target_spans[i, 1], context_embeddings.size(1)-1))
            
            if start_pos > end_pos:
                start_pos, end_pos = end_pos, start_pos
                
            span_rep = context_embeddings[i, start_pos:end_pos+1].mean(dim=0)
            span_reps.append(span_rep)
            
        span_reps = torch.stack(span_reps)
        
        # Enhanced context processing
        if self.use_context_attention:
            context_rep = self.context_attention(
                query=span_reps,
                context=context_embeddings,
                attention_mask=context_attention_mask
            )
        else:
            # Mean pooling with attention mask
            mask = context_attention_mask.unsqueeze(-1)
            context_rep = (context_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        return self.context_proj(context_rep)
        
        
    def forward(self, 
                word_input_ids: torch.Tensor,
                word_attention_mask: torch.Tensor,
                context_input_ids: torch.Tensor,
                context_attention_mask: torch.Tensor,
                target_spans: torch.Tensor):
        
        word_emb = self._encode_word(
            word_input_ids,
            word_attention_mask
        )
        context_emb = self._encode_context(
            context_input_ids,
            context_attention_mask,
            target_spans
        )
        word_emb_expanded = word_emb.unsqueeze(1)  # [batch, 1, hidden]
        context_emb_expanded = context_emb.unsqueeze(1)  
        combined = torch.cat([word_emb, context_emb], dim=-1)
        fused_emb, _ = self.fusion_attention(
            query=word_emb_expanded,
            key=context_emb_expanded,
            value=context_emb_expanded
        )
        fused_emb = self.fusion_dropout(fused_emb.squeeze(1))
        fused_emb = self.fusion_norm(fused_emb + word_emb)  
        output_emb = self.output_proj(fused_emb)

        return output_emb