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
import math

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
            gloss_id=sample["gloss_id"]
            self.all_samples.append({
                "sentence": sent,
                "target_word": target,
                "synset_id": sid,
                "gloss":gloss,
                "word_id":int(word_id),
                "gloss_id":gloss_id
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
        gid = s["gloss_id"]
        label = self.global_word_to_label[int(wid)]
        item = {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "word_id": label,
            "target_word": s["target_word"],
            "gloss" : s["gloss"],
            "synset_ids":sid,
            "gloss_id":gid
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
            gloss_id = torch.tensor([(b["gloss_id"]) for b in batch], dtype=torch.long)
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
                "gloss_id":gloss_id 
            }

        sentences = [b["sentence"] for b in batch]
        target_words = [b["target_word"] for b in batch]
        spans     = [b["target_span"] for b in batch]
        labels    = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
        glosses = [b["gloss"] for b in batch]
        gloss_id = torch.tensor([(b["gloss_id"]) for b in batch], dtype=torch.long)
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
            "gloss_id":gloss_id
        }


class ContrastiveBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, samples_per_gloss=2, max_gloss_per_batch=None, drop_last=False):
        """
        Args:
            dataset: Your PolyBERTtDataset instance
            batch_size: Total batch size
            samples_per_gloss: How many samples to include per gloss in a batch
            max_gloss_per_batch: Maximum number of glosses per batch (None for auto)
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.samples_per_gloss = samples_per_gloss
        
        # Auto-calculate max_gloss_per_batch if not specified
        self.max_gloss_per_batch = max_gloss_per_batch or (batch_size // samples_per_gloss)
        
        # Build indices organized by gloss_id and target_word
        self._build_indices()
        
    def _build_indices(self):
        # Create mapping from gloss_id to sample indices
        self.gloss_to_indices = defaultdict(list)
        
        # Also create mapping from target_word to gloss_ids
        self.word_to_glosses = defaultdict(set)
        
        for idx, sample in enumerate(self.dataset.all_samples):
            gloss_id = sample["gloss_id"]
            target_word = sample["target_word"]
            
            self.gloss_to_indices[gloss_id].append(idx)
            self.word_to_glosses[target_word].add(gloss_id)
        
        # Convert to regular dict and shuffle each gloss's indices
        self.gloss_to_indices = dict(self.gloss_to_indices)
        for gloss_id in self.gloss_to_indices:
            random.shuffle(self.gloss_to_indices[gloss_id])
        
        # Get all gloss_ids and shuffle them
        self.all_gloss_ids = list(self.gloss_to_indices.keys())
        random.shuffle(self.all_gloss_ids)
        
        # Track current position for each gloss
        self.gloss_ptr = {gloss_id: 0 for gloss_id in self.all_gloss_ids}
        
    def _get_samples_for_gloss(self, gloss_id, count):
        """Get next 'count' samples for given gloss_id"""
        indices = []
        remaining = count
        
        while remaining > 0:
            available = len(self.gloss_to_indices[gloss_id]) - self.gloss_ptr[gloss_id]
            if available == 0:
                break
                
            take = min(remaining, available)
            start = self.gloss_ptr[gloss_id]
            end = start + take
            indices.extend(self.gloss_to_indices[gloss_id][start:end])
            self.gloss_ptr[gloss_id] = end
            remaining -= take
        
        return indices
    
    def __iter__(self):
        # Reset gloss pointers
        self.gloss_ptr = {gloss_id: 0 for gloss_id in self.all_gloss_ids}
        random.shuffle(self.all_gloss_ids)
        
        # Create a list of all gloss_ids that still have samples remaining
        remaining_gloss_ids = [gloss_id for gloss_id in self.all_gloss_ids 
                              if self.gloss_ptr[gloss_id] < len(self.gloss_to_indices[gloss_id])]
        
        while len(remaining_gloss_ids) > 0:
            batch_indices = []
            
            # Strategy 1: Prioritize glosses that share target_words with other glosses in batch
            if len(batch_indices) > 0:
                # Get target_words already in batch
                batch_target_words = set()
                for idx in batch_indices:
                    batch_target_words.add(self.dataset.all_samples[idx]["target_word"])
                
                # Find gloss_ids that share these target_words but aren't already in batch
                candidate_gloss_ids = set()
                for word in batch_target_words:
                    for gloss_id in self.word_to_glosses[word]:
                        if (gloss_id in remaining_gloss_ids and 
                            gloss_id not in [self.dataset.all_samples[idx]["gloss_id"] for idx in batch_indices]):
                            candidate_gloss_ids.add(gloss_id)
                
                if candidate_gloss_ids:
                    selected_gloss = random.choice(list(candidate_gloss_ids))
                    samples = self._get_samples_for_gloss(selected_gloss, self.samples_per_gloss)
                    batch_indices.extend(samples)
            
            # Strategy 2: If batch still needs more samples, add new glosses
            while len(batch_indices) < self.batch_size and len(remaining_gloss_ids) > 0:
                # Select a gloss we haven't used yet in this batch
                available_gloss = [g for g in remaining_gloss_ids 
                                 if g not in [self.dataset.all_samples[idx]["gloss_id"] for idx in batch_indices]]
                
                if not available_gloss:
                    break
                
                selected_gloss = random.choice(available_gloss)
                samples = self._get_samples_for_gloss(selected_gloss, self.samples_per_gloss)
                batch_indices.extend(samples)
            
            # Update remaining gloss_ids
            remaining_gloss_ids = [gloss_id for gloss_id in self.all_gloss_ids 
                                  if self.gloss_ptr[gloss_id] < len(self.gloss_to_indices[gloss_id])]
            
            if len(batch_indices) < self.batch_size and self.drop_last:
                continue
            
            if batch_indices:
                yield batch_indices
    
    def __len__(self):
        # Approximate number of batches
        total_samples = sum(len(indices) for indices in self.gloss_to_indices.values())
        return total_samples // self.batch_size