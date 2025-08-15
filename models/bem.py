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

        # Average over target tokens
        target_mask = target_mask.float()
        lengths = target_mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled = (hidden_states * target_mask.unsqueeze(-1)).sum(dim=1) / lengths  # [B, H]
        return pooled

    def forward_gloss(self, input_ids, attention_mask):
        with torch.no_grad() if self.freeze_gloss else torch.enable_grad():
            outputs = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state  # [B, L, H]

        # Lấy CLS
        cls_vec = hidden_states[:, 0, :]  # [B, H]
        return cls_vec

    def batch_contrastive_loss(self, ctx_vecs, gloss_vecs, word_id):
        """
        ctx_vecs: [B, H]
        gloss_vecs: [B, H]
        word_id: [B] long tensor
        """
        sim = torch.matmul(ctx_vecs, gloss_vecs.T)  # [B, B]
        P = F.softmax(sim, dim=1)

        word_id_row = word_id.unsqueeze(0)
        word_id_col = word_id.unsqueeze(1)
        pos_mask = (word_id_row == word_id_col).float()

        pos_probs = (P * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
        loss = -torch.log(pos_probs + 1e-8).mean()

        return loss, sim

    def forward(self, context_inputs, gloss_inputs, target_mask):
        ctx_vecs = self.forward_context(
            context_inputs["input_ids"],
            context_inputs["attention_mask"],
            target_mask
        )
        gloss_vecs = self.forward_gloss(
            gloss_inputs["input_ids"],
            gloss_inputs["attention_mask"]
        )
        return ctx_vecs, gloss_vecs

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        config = {
            "encoder_name": self.encoder_name,
            "freeze_context": self.freeze_context,
            "freeze_gloss": self.freeze_gloss,
            "tie_encoders": self.tie_encoders
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
            tokenizer = AutoTokenizer.from_pretrained(save_directory)
        model = cls(
            encoder_name=config["encoder_name"],
            freeze_context=config["freeze_context"],
            freeze_gloss=config["freeze_gloss"],
            tie_encoders=config["tie_encoders"],
            tokenizer=tokenizer
        )
        state_dict = torch.load(os.path.join(save_directory, "pytorch_model.bin"), map_location="cpu")
        model.load_state_dict(state_dict)
        return model
