import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class PolyBERT(nn.Module):
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

    def forward_context(self, input_ids, attention_mask, target_idx):
        outputs = self.context_encoder(input_ids=input_ids, attention_mask=attention_mask)
        EC = outputs.last_hidden_state  # [B, L, H]

        batch_size = EC.size(0)
        r_wt = EC[torch.arange(batch_size), target_idx]  # [B, H]

        Q = r_wt.unsqueeze(1).repeat(1, self.polym, 1)  # [B, polym, H]
        K, V = EC, EC

        fused, _ = self.attn(Q, K, V)  # [B, polym, H]
        return fused

    def forward_gloss(self, input_ids, attention_mask):
        outputs = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask)
        EG = outputs.last_hidden_state  # [B, L, H]
        rg = EG[:, 0, :]  # CLS token [B, H]

        rFg = rg.unsqueeze(1).repeat(1, self.polym, 1)  # [B, polym, H]
        return rFg

    def batch_contrastive_loss(self, rF_wt, rF_g):
        rF_wt_mean = rF_wt.mean(dim=1)  # [B, H]
        rF_g_mean = rF_g.mean(dim=1)    # [B, H]
        MF = torch.matmul(rF_wt_mean, rF_g_mean.T)  # [B, B]
        P = F.softmax(MF, dim=1)
        Pd = P[torch.arange(P.size(0)), torch.arange(P.size(0))]
        loss = -torch.log(Pd).mean()
        return loss, MF

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
        loss, MF = self.batch_contrastive_loss(rF_wt, rF_g)
        return loss, MF

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
