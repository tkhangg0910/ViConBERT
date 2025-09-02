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
from .base_model import MLPBlock
    

class PolyBERT(nn.Module):
    def __init__(self, 
                tokenizer,
                bert_model_name: str = "vinai/phobert-base",
                cache_dir: str = "embeddings/base_models",
                hidden_dim: int = 512,
                out_dim: int = 768,
                dropout: float = 0.1,
                num_layers: int = 1,
                num_head: int = 3,
                context_window_size: int = 3,
                freeze_context: bool = False,
                freeze_gloss: bool = False,
                tie_encoders: bool = False,
                use_projection: bool = True):
        super().__init__()
        
        self.config = {
            "base_model": bert_model_name,
            "base_model_cache_dir": cache_dir,
            "hidden_dim": hidden_dim,
            "out_dim": out_dim,
            "dropout": dropout,
            "num_layers": num_layers,
            "num_head": num_head,
            "context_window_size": context_window_size,
            "freeze_context": freeze_context,
            "freeze_gloss": freeze_gloss,
            "tie_encoders": tie_encoders,
            "use_projection": use_projection
        }
        
        self.tokenizer = tokenizer
        self.freeze_context = freeze_context
        self.freeze_gloss = freeze_gloss
        self.tie_encoders = tie_encoders
        self.context_window_size = context_window_size
        self.use_projection = use_projection

        # Context encoder
        self.context_encoder = AutoModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
        self.context_encoder.resize_token_embeddings(len(tokenizer))
        
        # Gloss encoder
        if tie_encoders:
            self.gloss_encoder = self.context_encoder
        else:
            self.gloss_encoder = AutoModel.from_pretrained(bert_model_name, cache_dir=cache_dir)
            self.gloss_encoder.resize_token_embeddings(len(tokenizer))

        # Projection layers (optional)
        if use_projection:
            self.context_projection = MLPBlock(
                self.context_encoder.config.hidden_size,
                hidden_dim,
                out_dim,
                dropout=dropout,
                num_layers=num_layers
            )
            
            if not tie_encoders:
                self.gloss_projection = MLPBlock(
                    self.gloss_encoder.config.hidden_size,
                    hidden_dim,
                    out_dim,
                    dropout=dropout,
                    num_layers=num_layers
                )
            else:
                self.gloss_projection = self.context_projection
        
        self.context_attention = nn.MultiheadAttention(
            self.context_encoder.config.hidden_size,
            num_heads=num_head,
            dropout=dropout
        )
   

    def forward_context(self, input_ids, attention_mask, target_span):
        """
        Forward pass for context encoding
        Args:
            input_ids: [B, L] - tokenized context
            attention_mask: [B, L] - attention mask
            target_mask: [B, L] - mask for target tokens (for simple pooling)
            target_span: [B, 2] - start/end positions for span-based methods
        """
        with torch.no_grad() if self.freeze_context else torch.enable_grad():
            context_emb = self._encode_context_attentive(
                {"input_ids": input_ids, "attention_mask": attention_mask}, 
                target_span
            )
        # Apply projection if enabled
        if self.use_projection:
            context_emb = self.context_projection(context_emb.squeeze(0))
        else:
            context_emb = context_emb.squeeze(0)
            
        # L2 normalize
        context_emb = F.normalize(context_emb, p=2, dim=-1)
        return context_emb

    def forward_gloss(self, input_ids, attention_mask):
        """
        Forward pass for gloss encoding
        Args:
            input_ids: [B, L] - tokenized gloss/definition
            attention_mask: [B, L] - attention mask
        """
        with torch.no_grad() if self.freeze_gloss else torch.enable_grad():
            outputs = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, L, H]

        # Use CLS token for gloss representation
        gloss_emb = hidden_states[:, 0, :]  # [B, H]
        
        # Apply projection if enabled
        if self.use_projection:
            gloss_emb = self.gloss_projection(gloss_emb)
            
        # L2 normalize
        gloss_emb = F.normalize(gloss_emb, p=2, dim=-1)
        return gloss_emb

    def _encode_context_attentive(self, text, target_span):
        """Attentive context encoding"""
        outputs = self.context_encoder(**text)
        start_pos = target_span[:, 0]
        end_pos = target_span[:, 1]
        
        hidden_states = outputs[0]  
        positions = torch.arange(hidden_states.size(1), device=hidden_states.device)  
        
        mask = (positions >= start_pos.unsqueeze(1)) & (positions <= end_pos.unsqueeze(1))  
        masked_states = hidden_states * mask.unsqueeze(-1) 
        span_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1)  
        pooled_embeddings = masked_states.sum(dim=1) / span_lengths
        
        Q_value = pooled_embeddings.unsqueeze(0)  
        KV_value = hidden_states.permute(1, 0, 2)  
        context_emb, _ = self.context_attention(Q_value, KV_value, KV_value)
        
        return context_emb
        
    def forward(self, context_input=None, gloss_input=None, target_span=None, target_mask=None):
        """
        Forward pass - can handle both context and gloss encoding
        """
        
        if context_input is not None:
            context_emb = self.forward_context(
                input_ids=context_input.get("input_ids"),
                attention_mask=context_input.get("attention_mask"),
                target_mask=target_mask,
                target_span=target_span
            )
            
        if gloss_input is not None:
            gloss_emb = self.forward_gloss(
                input_ids=gloss_input.get("input_ids"),
                attention_mask=gloss_input.get("attention_mask")
            )
            
        return context_emb,gloss_emb
    
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
            bert_model_name=config["base_model"],
            cache_dir=config["base_model_cache_dir"],
            hidden_dim=config["hidden_dim"],
            out_dim=config["out_dim"],
            dropout=config["dropout"],
            num_layers=config["num_layers"],
            num_head=config["num_head"],
            context_window_size=config["context_window_size"],
            freeze_context=config.get("freeze_context", False),
            freeze_gloss=config.get("freeze_gloss", False),
            tie_encoders=config.get("tie_encoders", False),
            use_projection=config.get("use_projection", True)
        )

        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=torch.device('cpu')
        )
        
        model.load_state_dict(state_dict)
        return model