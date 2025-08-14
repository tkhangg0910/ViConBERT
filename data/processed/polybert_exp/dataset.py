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
    def __init__(self, samples, tokenizer, val_mode = False):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)
        self.val_mode =val_mode
        if self.val_mode:
            self.targetword2glosses = defaultdict(list)
        word_set = set()
        self.all_samples = []
        for sample in samples:
            sent = text_normalize(sample["sentence"])
            word_id = sample["word_id"]
            target = sample["target_word"]
            sid = sample["synset_id"]
            gloss=sample["gloss"]
            self.all_samples.append({
                "sentence": sent,
                "target_word": target,
                "synset_id": sid,
                "gloss":gloss,
                "word_id":int(word_id)
            })
            
            if self.val_mode:
                if gloss not in self.targetword2glosses[target]:
                    self.targetword2glosses[target].append(gloss)

            word_set.add(int(word_id))
            
        sorted_wids = sorted(word_set)
        self.global_word_to_label = {wid: i for i, wid in enumerate(sorted_wids)}

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
        wid = s["word_id"]
        label = self.global_word_to_label[int(wid)]
        item = {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "word_id": label,
            "target_word": s["target_word"],
            "gloss" : s["gloss"],
            "synset_ids":sid
        }
        if self.val_mode:
            # candidate glosses = all glosses observed for this target_word
            item["candidate_glosses"] = list(self.targetword2glosses[s["target_word"]])
        return item

    def collate_fn(self, batch):
        if self.val_mode:
            # batch: list of items (each item has candidate_glosses list)
            sentences = [b["sentence"] for b in batch]          # B
            spans     = [b["target_span"] for b in batch]       # B x (2)
            synset_ids = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
            word_id = torch.tensor([(b["word_id"]) for b in batch], dtype=torch.long)
            target_words = [b["target_word"] for b in batch]
            gold_glosses = [b["gloss"] for b in batch]

            # tokenise contexts (B)
            c_toks = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            )

            candidate_glosses = [b["candidate_glosses"] for b in batch]
        
            return {
                "context_input_ids": c_toks["input_ids"],           # [B, Lc]
                "context_attn_mask": c_toks["attention_mask"],      # [B, Lc]
                "target_spans": torch.tensor(spans, dtype=torch.long),  # [B,2]
                "synset_ids": synset_ids,
                "word_id": word_id,
                "target_words": target_words,
                "gold_glosses": gold_glosses,                
                "candidate_glosses": candidate_glosses, 
            }

        sentences = [b["sentence"] for b in batch]
        target_words = [b["target_word"] for b in batch]
        spans     = [b["target_span"] for b in batch]
        labels    = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
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
            "word_id":word_id,
            "target_words":target_words,
            "gloss_input_ids": g_tokes["input_ids"],
            "gloss_attn_mask": g_tokes["attention_mask"],
        }
from collections import defaultdict, deque
from math import ceil

class ContrastiveBatchSampler(Sampler):
    """
    BatchSampler that:
      - Avoids (as much as possible) putting two samples with same gloss in same batch.
      - Encourages that each sample has at least one batch-mate with the same target_word but different gloss.
      - Yields batches of indices.
    """
    def __init__(self, dataset, batch_size, seed=42):
        """
        dataset: object with dataset.all_samples where each item has 'target_word' and 'gloss'.
        batch_size: int, number of samples per batch.
        seed: for reproducibility.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.num_samples = len(dataset)
        
        # Build maps
        self.gloss_to_indices = defaultdict(list)
        self.target_to_glosses = defaultdict(set)
        self.gloss_to_target = {}
        
        for idx, s in enumerate(dataset.all_samples):
            g = s["gloss"]
            t = s["target_word"]
            self.gloss_to_indices[g].append(idx)
            self.target_to_glosses[t].add(g)
            self.gloss_to_target[g] = t

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def __iter__(self):
        rnd = random.Random(self.seed)
        all_indices = list(range(self.num_samples))
        rnd.shuffle(all_indices)
        
        # Shuffle indices per gloss so we don't always pick same samples
        gloss_queues = {g: list(idxs) for g, idxs in self.gloss_to_indices.items()}
        for g in gloss_queues:
            rnd.shuffle(gloss_queues[g])
        
        available_glosses = set(gloss_queues.keys())
        batches = []
        
        while any(gloss_queues.values()):
            batch = []
            used_glosses = set()
            
            # Step 1: pick a random target that still has >= 2 glosses available
            candidate_targets = [t for t, glosses in self.target_to_glosses.items()
                                 if len(glosses & available_glosses) >= 2]
            if candidate_targets:
                t = rnd.choice(candidate_targets)
                glosses_avail = list(self.target_to_glosses[t] & available_glosses)
                rnd.shuffle(glosses_avail)
                for g in glosses_avail:
                    if len(batch) >= self.batch_size:
                        break
                    if gloss_queues[g]:
                        batch.append(gloss_queues[g].pop())
                        used_glosses.add(g)
            
            # Step 2: fill remaining slots with other glosses (avoid duplicates)
            remaining_slots = self.batch_size - len(batch)
            if remaining_slots > 0:
                other_glosses = list(available_glosses - used_glosses)
                rnd.shuffle(other_glosses)
                for g in other_glosses:
                    if remaining_slots <= 0:
                        break
                    if gloss_queues[g]:
                        batch.append(gloss_queues[g].pop())
                        used_glosses.add(g)
                        remaining_slots -= 1
            
            # Step 3: if still not full, allow repeating gloss (fallback)
            if len(batch) < self.batch_size:
                leftover_glosses = [g for g, q in gloss_queues.items() if q]
                rnd.shuffle(leftover_glosses)
                for g in leftover_glosses:
                    if len(batch) >= self.batch_size:
                        break
                    if gloss_queues[g]:
                        batch.append(gloss_queues[g].pop())
            
            # Update availability
            for g in list(available_glosses):
                if not gloss_queues[g]:
                    available_glosses.remove(g)
            
            batches.append(batch)
        
        # Shuffle batches order
        rnd.shuffle(batches)
        
        for b in batches:
            yield b
