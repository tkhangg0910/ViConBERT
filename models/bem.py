import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class BiEncoderModel(nn.Module):
    def __init__(self, encoder_name="vinai/phobert-base", freeze_context=False, freeze_gloss=False, tie_encoders=False, tokenizer=None):
        super().__init__()
        self.encoder_name = encoder_name
        self.freeze_context = freeze_context
        self.freeze_gloss = freeze_gloss
        self.tie_encoders = tie_encoders
        self.tokenizer = tokenizer

        # Context encoder
        self.context_encoder = AutoModel.from_pretrained(encoder_name)
        # Gloss encoder
        if tie_encoders:
            self.gloss_encoder = self.context_encoder
        else:
            self.gloss_encoder = AutoModel.from_pretrained(encoder_name)

        self.hidden_size = self.context_encoder.config.hidden_size

    def forward_context(self, input_ids, attention_mask, target_mask):
        """
        input_ids: [B, L]
        attention_mask: [B, L]
        target_mask: [B, L] 
        """
        with torch.no_grad() if self.freeze_context else torch.enable_grad():
            outputs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, L, H]

        # Average over target tokens - SAME AS ORIGINAL
        target_mask = target_mask.float()
        lengths = target_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (hidden_states * target_mask.unsqueeze(-1)).sum(dim=1) / lengths  # [B, H]
        
        # L2 normalize như code gốc
        pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def forward_gloss(self, input_ids, attention_mask):
        with torch.no_grad() if self.freeze_gloss else torch.enable_grad():
            outputs = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, L, H]

        # Lấy CLS như code gốc
        cls_vec = hidden_states[:, 0, :]  # [B, H]
        
        # L2 normalize như code gốc
        cls_vec = F.normalize(cls_vec, p=2, dim=-1)
        return cls_vec