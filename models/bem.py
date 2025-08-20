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
    
    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)

        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

        # Save config
        config = {
            "encoder_name": self.encoder_name,
            "freeze_context": self.freeze_context,
            "freeze_gloss": self.freeze_gloss,
            "tie_encoders": self.tie_encoders
        }
        with open(os.path.join(save_directory, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        # Save tokenizer if available
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, save_directory, tokenizer=None):
        # Load config
        with open(os.path.join(save_directory, "config.json"), "r", encoding="utf-8") as f:
            config = json.load(f)

        # Load tokenizer if not provided
        if tokenizer is None:
            try:
                tokenizer = AutoTokenizer.from_pretrained(save_directory)
            except:
                tokenizer = None

        # Khởi tạo model
        model = cls(
            encoder_name=config["encoder_name"],
            freeze_context=config.get("freeze_context", False),
            freeze_gloss=config.get("freeze_gloss", False),
            tie_encoders=config.get("tie_encoders", False),
            tokenizer=tokenizer
        )

        # Load weights
        state_dict = torch.load(
            os.path.join(save_directory, "pytorch_model.bin"),
            map_location=torch.device("cpu")
        )
        model.load_state_dict(state_dict)
        return model

