import random
import re
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize
from torch.utils.data import Sampler
from collections import defaultdict

class PolyBERTtDataset(Dataset):
    def __init__(self, samples, tokenizer, ):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)

        synset_set = set()
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
            synset_set.add(sid)
        sorted_sids = sorted(synset_set)
        self.global_synset_to_label = {sid: i for i, sid in enumerate(sorted_sids)}

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
        label = self.global_synset_to_label[sid]
        return {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "synset_id": label,
            "gloss": s["gloss"]
        }

    def collate_fn(self, batch):
        sentences = [b["sentence"] for b in batch]
        spans     = [b["target_span"] for b in batch]
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
    Batch sampler for WSD contrastive learning:
    - Prioritize same target_word
    - No duplicate (word_id, synset_id) in batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Group indices by target_word
        self.target2indices = defaultdict(list)
        for idx, item in enumerate(dataset):
            self.target2indices[item['target_word']].append(idx)
    
    def __iter__(self):
        all_targets = list(self.target2indices.keys())
        random.shuffle(all_targets)
        
        for target_word in all_targets:
            indices = self.target2indices[target_word].copy()
            random.shuffle(indices)
            
            batch = []
            used_pairs = set()
            
            for idx in indices:
                item = self.dataset[idx]
                pair = (item['word_id'], item['synset_id'])
                if pair in used_pairs:
                    continue
                batch.append(idx)
                used_pairs.add(pair)
                
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
                    used_pairs = set()
            
            # leftover
            if batch:
                yield batch
    
    def __len__(self):
        return sum(len(indices) for indices in self.target2indices.values()) // self.batch_size
