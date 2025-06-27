import logging
import os
import json
from transformers import AutoTokenizer
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
        context_window_size:int = 3
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
        self.config = {
            "base_model": model_name,
            "base_model_cache_dir": cache_dir,
            "model": {
                "fusion_hidden_dim": fusion_hidden_dim,
                "dropout": dropout,
                "freeze_base": freeze_base,
                "fusion_num_layers": fusion_num_layers,
                "wp_num_layers": wp_num_layers,
                "cp_num_layers": cp_num_layers,
                "context_window_size": context_window_size
            }
        }

        self.base_model = AutoModel.from_pretrained(model_name, 
                                            cache_dir=cache_dir)

        
    def forward(self, 
                word_input_ids: torch.Tensor,
                word_attention_mask: torch.Tensor,
                context_input_ids: torch.Tensor,
                context_attention_mask: torch.Tensor,
                target_spans: torch.Tensor):
    
        return fused_embed
    
    def save_pretrained(self, save_directory):
        """Save pretrained Hugging Face model"""
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
        config.update({"add_pooling_layer": False})
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(save_directory)
            except:
                raise ValueError("Không tìm thấy tokenizer trong thư mục")
         
        model = cls(
            tokenizer=tokenizer,
            model_name=config["base_model"],
            cache_dir=config["base_model_cache_dir"],
            fusion_hidden_dim=config["model"]["fusion_hidden_dim"],
            dropout=config["model"]["dropout"],
            freeze_base=config["model"]["freeze_base"],
            fusion_num_layers=config["model"]["fusion_num_layers"],
            wp_num_layers=config["model"]["wp_num_layers"],
            cp_num_layers=config["model"]["cp_num_layers"],
            context_window_size=config["model"]["context_window_size"]
        )
        
        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(state_dict)
        return model
    