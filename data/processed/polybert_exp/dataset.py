import random
import re
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import PreTrainedTokenizerFast
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor,SentenceMasking
from utils.process_data import text_normalize
import math
from torch.utils.data import Sampler

class PolyBERTtDataset(Dataset):
    def __init__(self, gloss_embeddings_path, samples, tokenizer, use_sent_masking=False):
        self.tokenizer = tokenizer
        self.use_sent_masking = use_sent_masking
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)

        # Lưu samples thô
        self.all_samples = []
        synset_set = set()
        for sample in samples:
            sent = text_normalize(sample["sentence"])
            target = sample["target_word"]
            sid = sample["synset_id"]
            self.all_samples.append({
                "sentence": sent,
                "target_word": target,
                "synset_id": sid
            })
            synset_set.add(sid)

        # Xây mapping synset_id <-> label (contiguous từ 0)
        sorted_sids = sorted(synset_set)
        self.global_synset_to_label = {sid: i for i, sid in enumerate(sorted_sids)}
        self.global_label_to_synset = {i: sid for i, sid in enumerate(sorted_sids)}

        # Precompute spans (hoặc masked sents)
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

        # Load gloss embeddings: dict synset_id -> tensor
        self.gloss_embeddings = torch.load(gloss_embeddings_path)

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s = self.all_samples[idx]
        sid = s["synset_id"]
        label = self.global_synset_to_label[sid]
        return {
            "sentence": self.span_indices[idx] if self.use_sent_masking else s["sentence"],
            "target_span": None if self.use_sent_masking else self.span_indices[idx],
            "synset_id": sid,
            "synset_ids": label,         
            "gloss_embd": self.gloss_embeddings[sid]
        }

    def collate_fn(self, batch):
        sentences = [b["sentence"] for b in batch]
        spans     = [b["target_span"] for b in batch] if not self.use_sent_masking else None
        labels    = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
        glosses   = torch.stack([b["gloss_embd"] for b in batch])

        toks = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        return {
            "context_input_ids": toks["input_ids"],
            "context_attn_mask": toks["attention_mask"],
            "target_spans": torch.tensor(spans, dtype=torch.long) if spans else None,
            "synset_ids": labels,
            "gloss_embd": glosses
        }



class SynsetBatchSampler(Sampler):
    def __init__(self, labels, batch_size, min_per_class=4, shuffle=True):
        self.labels = labels
        self.batch_size = batch_size
        self.min_per_class = min_per_class
        self.shuffle = shuffle

        from collections import defaultdict
        self.lab2idx = defaultdict(list)
        for idx, lab in enumerate(labels):
            self.lab2idx[lab].append(idx)

        self.label_groups = []
        for lab, idxs in self.lab2idx.items():
            if len(idxs) >= min_per_class:
                self.label_groups.append((lab, idxs))

    def __iter__(self):
        all_samples = []

        if self.shuffle:
            random.shuffle(self.label_groups)

        for lab, idxs in self.label_groups:
            random.shuffle(idxs)
            # Group into pairs (or more)
            for i in range(0, len(idxs) - self.min_per_class + 1, self.min_per_class):
                group = idxs[i:i+self.min_per_class]
                if len(group) == self.min_per_class:
                    all_samples.append(group)

        # Flatten and batch
        random.shuffle(all_samples)
        flat = [i for g in all_samples for i in g]
        batches = [flat[i:i+self.batch_size] for i in range(0, len(flat), self.batch_size)]

        for b in batches:
            yield b

    def __len__(self):
        return math.ceil(len(self.labels) / self.batch_size)