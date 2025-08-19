import random
import re
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize
from collections import defaultdict

class BEMDataset(Dataset):
    def __init__(self, samples, tokenizer, train_gloss_size=256, val_mode=False):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)
        self.val_mode = val_mode
        self.train_gloss_size = train_gloss_size

        self.targetword2glosses = defaultdict(list)
        self.gloss_to_id = {}  # Map gloss text to ID
        
        word_set = set()
        self.all_samples = []
        gloss_set = set()

        for sample in samples:
            sent = text_normalize(sample["sentence"])
            word_id = sample["word_id"]
            target = sample["target_word"]
            sid = sample["synset_id"]
            gloss = sample["gloss"]
            gloss_id = sample["gloss_id"]

            self.all_samples.append({
                "sentence": sent,
                "target_word": target,
                "synset_id": sid,
                "gloss": gloss,
                "word_id": int(word_id),
                "gloss_id": gloss_id
            })

            if gloss not in self.targetword2glosses[target]:
                self.targetword2glosses[target].append(gloss)
            
            # FIXED: Build proper gloss mapping
            self.gloss_to_id[gloss] = gloss_id
            
            word_set.add(int(word_id))
            gloss_set.add(gloss)

        sorted_wids = sorted(word_set)
        self.global_word_to_label = {wid: i for i, wid in enumerate(sorted_wids)}
        self.global_gloss_pool = list(gloss_set)

        self.span_indices = []
        for s in tqdm(self.all_samples, desc="Computing spans", ascii=True):
            idxs = self.span_extractor.get_span_indices(
                s["sentence"], s["target_word"]
            )
            self.span_indices.append(idxs or (0, 0))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s = self.all_samples[idx]
        sid = s["synset_id"]
        wid = s["word_id"]
        gid = s["gloss_id"]
        label = self.global_word_to_label[int(wid)]
        
        gold_gloss = s["gloss"]
        target_word = s["target_word"]
        
        # Get all possible glosses for this target word
        all_candidates = list(set(self.targetword2glosses[target_word]))
        
        # Ensure gold gloss is always included
        if gold_gloss not in all_candidates:
            all_candidates.append(gold_gloss)
        
        item = {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "word_id": label,
            "target_word": target_word,
            "gold_gloss": gold_gloss,
            "synset_ids": sid,
            "gloss_id": gid,
            "candidate_glosses": all_candidates
        }

        return item

    def collate_fn(self, batch):
        sentences = [b["sentence"] for b in batch]
        gold_glosses = [b["gold_gloss"] for b in batch]
        spans = [b["target_span"] for b in batch]
        synset_ids = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
        target_words = [b["target_word"] for b in batch]
        gloss_ids = torch.tensor([b["gloss_id"] for b in batch], dtype=torch.long)

        # Tokenize contexts
        c_toks = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True
        )

        # Handle candidate glosses
        final_candidates = []
        gold_indices = []  # Store gold position in each candidate list
        
        for i, b in enumerate(batch):
            gold_gloss = b["gold_gloss"]
            candidates = b["candidate_glosses"].copy()
            
            if not self.val_mode:
                # Training: sample random glosses but ensure gold is included
                if len(candidates) > self.train_gloss_size:
                    # Keep gold + random sample of others
                    others = [c for c in candidates if c != gold_gloss]
                    sampled_others = random.sample(others, self.train_gloss_size - 1)
                    candidates = [gold_gloss] + sampled_others
                elif len(candidates) < self.train_gloss_size:
                    # Add random glosses from global pool
                    needed = self.train_gloss_size - len(candidates)
                    available = [g for g in self.global_gloss_pool if g not in candidates]
                    if len(available) >= needed:
                        extra = random.sample(available, needed)
                        candidates.extend(extra)
            
            # Shuffle candidates but remember gold position
            random.shuffle(candidates)
            try:
                gold_idx = candidates.index(gold_gloss)
            except ValueError:
                # Fallback: put gold at index 0
                candidates = [gold_gloss] + [c for c in candidates if c != gold_gloss]
                gold_idx = 0
            
            final_candidates.append(candidates)
            gold_indices.append(gold_idx)

        return {
            "context_input_ids": c_toks["input_ids"],
            "context_attn_mask": c_toks["attention_mask"],
            "target_spans": torch.tensor(spans, dtype=torch.long),
            "synset_ids": synset_ids,
            "word_id": word_id,
            "gold_glosses": gold_glosses,
            "target_words": target_words,
            "candidate_glosses": final_candidates,
            "gloss_ids": gloss_ids,
            "gold_indices": torch.tensor(gold_indices, dtype=torch.long) 
        }
