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
    


class PolyBERT_Original(nn.Module):
    def __init__(self, bert_model_name="vinai/phobert-base", polym=64, num_heads=8, tokenizer=None):
        super().__init__()
        self.bert_model_name = bert_model_name
        self.polym = polym
        self.num_heads = num_heads
        self.tokenizer = tokenizer

        self.context_encoder = AutoModel.from_pretrained(bert_model_name)
        self.gloss_encoder = AutoModel.from_pretrained(bert_model_name)

        hidden_size = self.context_encoder.config.hidden_size
        self.attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=num_heads, batch_first=True)

    def forward_context(self, input_ids, attention_mask, target_span):
        """
        input_ids: Tensor [B, L]
        attention_mask: Tensor [B, L]
        target_span: Tensor [B, 2] with (start_idx, end_idx) inclusive
        returns: fused context embeddings [B, polym, H]
        """
        # encode context
        outputs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, L, H]

        # extract span positions
        start_pos = target_span[:, 0]  # [B]
        end_pos   = target_span[:, 1]  # [B]

        # build mask for spans: [B, L]
        seq_len = hidden_states.size(1)
        positions = torch.arange(seq_len, device=hidden_states.device).unsqueeze(0)  # [1, L]
        mask = (positions >= start_pos.unsqueeze(1)) & (positions <= end_pos.unsqueeze(1))  # [B, L]

        # zero-out non-span states and average-pool over span
        masked_states = hidden_states * mask.unsqueeze(-1)  # [B, L, H]
        span_lengths = mask.sum(dim=1, keepdim=True).clamp(min=1).to(hidden_states.dtype)  # [B, 1]
        pooled_embeddings = masked_states.sum(dim=1) / span_lengths  # [B, H]

        # create queries by replicating pooled span embedding polym times
        Q = pooled_embeddings.unsqueeze(1).repeat(1, self.polym, 1)  # [B, polym, H]
        K = hidden_states  # [B, L, H]
        V = hidden_states  # [B, L, H]

        # perform multi-head attention (batch_first=True)
        fused, _ = self.attn(Q, K, V)  # [B, polym, H]

        return fused
    def contrastive_classification_loss(self, logits, label):
        """
        rF_wt: [B, polym, H]  - context embeddings
        rF_g:  [B, polym, H]  - gloss embeddings
        label: [B] long tensor - (0..B-1).
        """
        # CrossEntropyLoss trực tiếp
        loss = F.cross_entropy(logits, label)

        return loss, logits

    def forward_gloss(self, input_ids, attention_mask):
        outputs = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask)
        EG = outputs.last_hidden_state  # [B, L, H]
        rg = EG[:, 0, :]  # CLS token [B, H]

        rFg = rg.unsqueeze(1).repeat(1, self.polym, 1)  # [B, polym, H]
        return rFg

    def batch_contrastive_loss_with_id(self, rF_wt, rF_g, word_id, temperature=0.05):
        B = rF_wt.size(0)
        # Flatten and normalize vectors
        rF_wt_flat = F.normalize(rF_wt.reshape(B, -1), p=2, dim=1)
        rF_g_flat = F.normalize(rF_g.reshape(B, -1), p=2, dim=1)
        
        # Compute similarity matrix with temperature scaling
        sim = torch.mm(rF_wt_flat, rF_g_flat.t()) / temperature
        
        # Create positive mask (same word_id)
        pos_mask = word_id.unsqueeze(0) == word_id.unsqueeze(1)
        pos_mask = pos_mask.to(sim.device)
        
        # Compute logits for positive pairs
        pos_sim = sim.masked_select(pos_mask).view(B, -1)
        
        # Compute logits for negative pairs
        neg_sim = sim.masked_select(~pos_mask).view(B, -1)
        
        # Concatenate for cross entropy: positive is always index 0
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # Targets: positive at index 0
        targets = torch.zeros(B, dtype=torch.long, device=sim.device)
        
        loss = F.cross_entropy(logits, targets)
        return loss, sim
    def batch_contrastive_loss(self, rF_wt, rF_g, temperature=0.07):
        B = rF_wt.size(0)
        rF_wt_flat = F.normalize(rF_wt.reshape(B, -1), p=2, dim=1)
        rF_g_flat  = F.normalize(rF_g.reshape(B, -1), p=2, dim=1)

        sim_matrix = torch.matmul(rF_wt_flat, rF_g_flat.T) / temperature

        targets = torch.arange(B).to(rF_wt.device)
        loss = F.cross_entropy(sim_matrix, targets)

        return loss, sim_matrix




    def forward(self, context_inputs, gloss_inputs, target_idx):
        rF_wt = self.forward_context(
            context_inputs["input_ids"],
            context_inputs["attention_mask"],
            target_idx
        )
        rF_g = self.forward_gloss(
            gloss_inputs["input_ids"],
            gloss_inputs["attention_mask"]
        )
        return rF_wt, rF_g

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        config = {
            "bert_model_name": self.bert_model_name,
            "polym": self.polym,
            "num_heads": self.num_heads
        }
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory, tokenizer=None):
        with open(os.path.join(save_directory, "config.json"), "r") as f:
            config = json.load(f)

        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(save_directory)
            except:
                raise ValueError("Cannot fine tokenizer")

        model = cls(
            bert_model_name=config["bert_model_name"],
            polym=config["polym"],
            num_heads=config["num_heads"],
            tokenizer=tokenizer
        )

        # Load trọng số
        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=torch.device('cpu')
        )
        model.load_state_dict(state_dict)
        return model