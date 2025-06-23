import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PreTrainedTokenizerFast
from typing import List, Optional, Tuple, Dict
from utils.span_extractor import SpanExtractor, create_masked_version
from typing import Callable

class MLPBlock(nn.Module):
    """Enhanced neural block to combine context and word representations"""
    
    def __init__(self, input_dim: int, hidden_dim: int, 
                 output_dim: int,num_layers: int = 2, dropout: float = 0.1,
                 activation: Callable = nn.GELU,use_residual: bool = True,
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
            x = self.final_activation()
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
                word_input: torch.Tensor, 
                context_input: torch.Tensor
                ):
        """
        Forward pass with flexible span representation
        """
        word_emb = self._encode_word(
            word_input["input_ids"],
            word_input["attention_mask"]
        )
        context_emb = self._encode_context(
            context_input["input_ids"],
            context_input["attention_mask"]
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
    
    def tokenize_with_target(self, 
                           texts: List[str], 
                           target_phrases: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimized tokenization with span index precomputation
        """
        
        word_input = self.tokenizer(
            target_phrases, 
            padding=True,
            truncation=True,
            max_length=128 ,
            return_tensors="pt"
        )
        masked_sents =  [create_masked_version(text, phrase, self.tokenizer) for text, phrase in zip(texts, target_phrases)]
        context_input = self.tokenizer(
            masked_sents, 
            padding=True,
            truncation=True,
            max_length=128 ,
            return_tensors="pt"
        )
        return word_input, context_input
    
    def encode(self, 
           texts: List[str], 
           target_phrases: List[str]) -> torch.Tensor:
        word_input, context_input = self.tokenize_with_target(texts, target_phrases)
        
        with torch.inference_mode():
            embeddings = self.forward(word_input, context_input)
        return embeddings

class SynoViSenseEmbeddingV2(nn.Module):
    def __init__(self, 
        tokenizer,
        model_name: str = "vinai/phobert-base",
        cache_dir: str ="embeddings/base_models",
        fusion_hidden_dim: int = 512,
        wp_num_layers:int=1,
        sp_num_layers:int=1,
        dropout: float = 0.1,
        freeze_base: bool = False,
        fusion_num_layers:int=1
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
        
        self.tokenizer = tokenizer
        self.base_model = AutoModel.from_pretrained(model_name, 
                                              cache_dir=cache_dir)
        self.hidden_size = self.base_model.config.hidden_size

        self.fusion_gate = MLPBlock(
            input_dim=self.hidden_size*2,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=fusion_num_layers,
            final_activation=nn.Sigmoid()
        )
        
        self.context_proj = MLPBlock(
            input_dim=self.hidden_size,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=sp_num_layers
        )
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.word_projection = MLPBlock(
            input_dim=self.hidden_size,
            hidden_dim=fusion_hidden_dim,
            output_dim=self.hidden_size,
            dropout=dropout,
            num_layers=wp_num_layers
        )
     
    def _encode_word(self, word_input_ids, word_attention_mask):
        outputs = self.base_model(
            input_ids=word_input_ids,
            attention_mask=word_attention_mask
        )
        word_rep = outputs.last_hidden_state[:, 0] 
        return self.word_projection(word_rep)
    
    
    # def forward(self, input_ids, attention_mask, span_positions):

        