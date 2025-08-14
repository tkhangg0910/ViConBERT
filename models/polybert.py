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

    def forward_gloss(self, input_ids, attention_mask):
        outputs = self.gloss_encoder(input_ids=input_ids, attention_mask=attention_mask)
        EG = outputs.last_hidden_state  # [B, L, H]
        rg = EG[:, 0, :]  # CLS token [B, H]

        rFg = rg.unsqueeze(1).repeat(1, self.polym, 1)  # [B, polym, H]
        return rFg

    def batch_contrastive_loss(self, rF_wt, rF_g, word_id):
        """
        rF_wt: [B, polym, H]  - context embeddings
        rF_g:  [B, polym, H]  - gloss embeddings
        word_id: [B] long tensor, global word_id
                samples with same word_id are positives
        """
        B = rF_wt.size(0)
        print(rF_wt.shape)
        print(rF_g.shape)
        rF_wt_flat = rF_wt.view(B, -1)  # [B, polym*H]
        rF_g_flat  = rF_g.view(B, -1)   # [B, polym*H]

        # similarity matrix [B, B]
        sim = torch.matmul(rF_wt_flat, rF_g_flat.T)

        # softmax over rows
        P = F.softmax(sim, dim=1)

        # positive mask [B, B]: same global word_id
        word_id_row = word_id.unsqueeze(0)  # [1, B]
        word_id_col = word_id.unsqueeze(1)  # [B, 1]
        pos_mask = (word_id_row == word_id_col).float()  # [B, B]

        # compute probability of positives
        pos_probs = (P * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1.0)
        loss = -torch.log(pos_probs + 1e-8).mean()

        return loss, sim


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
