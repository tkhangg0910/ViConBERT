import random
import re
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize
import math
from torch.utils.data import Sampler

class PolyBERTtDataset(Dataset):
    def __init__(self, samples, tokenizer, use_sent_masking=False):
        self.tokenizer = tokenizer
        self.use_sent_masking = use_sent_masking
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)

        self.all_samples = []
        for sample in samples:
            sent = text_normalize(sample["sentence"])
            target = sample["target_word"]
            sid = sample["synset_id"]
            gloss=sample["gloss"]
            self.all_samples.append({
                "sentence": sent,
                "target_word": target,
                "synset_id": sid,
                "gloss":gloss
            })

        self.span_indices = []
        for s in tqdm(self.all_samples, desc="Computing spans",ascii=True):
            idxs = self.span_extractor.get_span_indices(
                s["sentence"], s["target_word"]
            )
            # if idxs:
            #     pred = self.span_extractor.get_span_text_from_indices(s["sentence"],idxs)
            #     if s["target_word"].lower().strip()!= pred.lower().strip():
            #         print(f"sentence: {s['sentence']}")
            #         print(f"target: {s['target_word']}")
            #         print(f"pred: {pred}")
            self.span_indices.append(idxs or (0,0))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s = self.all_samples[idx]
        sid = s["synset_id"]
        return {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "synset_id": sid,
            "gloss": s["gloss"]
        }

    def collate_fn(self, batch):
        sentences = [b["sentence"] for b in batch]
        spans     = [b["target_span"] for b in batch] if not self.use_sent_masking else None
        labels    = torch.tensor([b["synset_id"] for b in batch], dtype=torch.long)
        glosses = [b["gloss"] for b in batch]

        c_toks = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        g_tokes = self.tokenizer(
            glosses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        return {
            "context_input_ids": c_toks["input_ids"],
            "context_attn_mask": c_toks["attention_mask"],
            "target_spans": torch.tensor(spans, dtype=torch.long) if spans else None,
            "synset_ids": labels,
            "gloss_input_ids": g_tokes["input_ids"],
            "gloss_attn_mask": g_tokes["attention_mask"],
        }


class ContrastiveBatchSampler(Sampler):
    """
    BatchSampler for contrastive learning in PolyBERT.
    Ensures each batch has aligned (context, gloss) pairs.
    """
    def __init__(self, dataset_size, batch_size, shuffle=True, drop_last=False):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.indices = list(range(dataset_size))

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)
        
        # yield batches
        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.dataset_size // self.batch_size
        else:
            return (self.dataset_size + self.batch_size - 1) // self.batch_size
