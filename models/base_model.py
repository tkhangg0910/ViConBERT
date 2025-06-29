import logging
import os
import json
from transformers import AutoTokenizer
import torch.nn.functional as F
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
                dropout: float = 0.3,
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
        x = self.output_layer(x)
        if self.final_activation:
            x = self.final_activation(x)

        
        return x
    
class ViSynoSenseEmbedding(nn.Module):
    def __init__(self, tokenizer,
        model_name: str = "vinai/phobert-base",
        cache_dir: str ="embeddings/base_models",
        hidden_dim: int = 512,
        out_dim:int = 768,
        dropout: float = 0.1,
        num_layers:int=1,
        num_head:int=3,
        polym = 8,
        encoder_type:str="attentive",
        context_window_size:int=3,
        use_proj:bool=True
        ):
        super().__init__()
        self.config = {
            "base_model": model_name,
            "base_model_cache_dir": cache_dir,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "dropout": dropout,
            "num_layers": num_layers,
            "num_head": num_head,
            "polym": polym,
            "encoder_type": encoder_type,
            "context_window_size": context_window_size,
        }
        self.tokenizer =tokenizer
        self.context_encoder = AutoModel.from_pretrained(model_name,cache_dir=cache_dir)
        self.context_encoder.resize_token_embeddings(len(tokenizer))
        self.polym=polym
        self.context_projection = MLPBlock(
            self.context_encoder.config.hidden_size,
            hidden_dim,
            out_dim,
            dropout=dropout,
            num_layers=num_layers
            
        )
        self.encoder_type =encoder_type
        self.context_attention=nn.MultiheadAttention(
            self.context_encoder.config.hidden_size,
            num_heads=num_head,
            dropout=dropout
        )
        self.context_window_size = context_window_size
        self.context_layer_weights = nn.Parameter(torch.ones(num_layers))


    def _encode_context_sep(self, text, target_spans):
        
        outputs = self.context_encoder(
            **text,
            output_hidden_states=True
        )
        
        all_hidden_states = outputs.hidden_states[1:]
        
        norm_weights = torch.softmax(self.context_layer_weights, dim=0)
        
        context_embeddings = torch.zeros_like(all_hidden_states[0])
        for i, hidden_state in enumerate(all_hidden_states):
            context_embeddings += norm_weights[i] * hidden_state
        
        batch_size, seq_len, _ = context_embeddings.shape
        positions = torch.arange(seq_len, device=context_embeddings.device).expand(batch_size, -1)
        
        start_pos = target_spans[:, 0]
        end_pos = target_spans[:, 1]
        center = (start_pos + end_pos) / 2
        dist = torch.abs(positions - center.unsqueeze(1))
        
        weights = 1.0 / (dist + 1.0)
        weights = torch.where(dist <= self.context_window_size, weights, torch.zeros_like(weights))
        weights = weights * text["context_attention_mask"].float()
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        
        weighted_emb = context_embeddings * weights.unsqueeze(-1)
        context_vectors = weighted_emb.sum(dim=1)
        
        return context_vectors
    
    def _encode_context_attentive(self, text, target_span):
        outputs = self.context_encoder(**text)
        start_pos = target_span[:, 0]
        end_pos = target_span[:, 1]
        
        hidden_states = outputs[0]  
        
        positions = torch.arange(hidden_states.size(1), device=hidden_states.device)  
        
        mask = (positions >= start_pos.unsqueeze(1)) & (positions <= end_pos.unsqueeze(1))  
        masked_states = hidden_states * mask.unsqueeze(-1) 
        span_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  
        pooled_embeddings = masked_states.sum(dim=1) / span_lengths
        
        Q_value = pooled_embeddings.unsqueeze(0).expand(self.polym,-1, -1)
        KV_value = hidden_states.permute(1, 0, 2)  
        context_emb, _ = self.context_attention(
                Q_value, KV_value, KV_value
        )
        return context_emb
        
    def forward(self, context, target_span):
        """Forward pass"""
        context_emb=  self._encode_context_attentive(context,target_span) if self.encoder_type=="attentive" else self._encode_context_sep(context,target_span)
        
        return self.context_projection(context_emb)
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config, f, indent=2)

        if hasattr(self, 'tokenizer'):
            self.tokenizer.save_pretrained(save_directory)


    @classmethod
    def from_pretrained(cls, save_directory, tokenizer=None):
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            config = json.load(f)

        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(save_directory)
            except:
                raise ValueError("Không tìm thấy tokenizer trong thư mục")

        model = cls(
            tokenizer=tokenizer,
            model_name=config["base_model"],
            cache_dir=config["base_model_cache_dir"],
            hidden_dim=config["hidden_dim"],
            out_dim=config["out_dim"],
            dropout=config["dropout"],
            num_layers=config["num_layers"],
            num_head=config["num_head"],
            polym=config["polym"],
            encoder_type=config["encoder_type"],
            context_window_size=config["context_window_size"],
            use_proj=config["use_proj"]
        )

        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=torch.device('cpu')
        )
        
        model.load_state_dict(state_dict)
        return model
