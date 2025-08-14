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
     - Avoids having two samples with the same gloss inside one batch (if possible).
     - Encourages sampling multiple items from same target_word but different glosses.
     - Yields lists of indices (batches). Use in DataLoader as `batch_sampler=...`.
    """

    def __init__(self, dataset, batch_size, prefer_same_target=True, seed=None):
        """
        dataset: your dataset object where dataset.all_samples[i] has keys "gloss" and "target_word"
        batch_size: int
        prefer_same_target: if True, sampler will try to add multiple glosses from the same target_word
        seed: optional RNG seed for reproducibility
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.prefer_same_target = prefer_same_target
        self.seed = seed if seed is not None else random.randint(0, 2**31-1)

        # Build index maps
        self.gloss_to_indices = defaultdict(list)   # gloss -> [idx,...]
        self.target_to_glosses = defaultdict(set)   # target -> set(gloss)
        self.gloss_to_target = {}

        for idx, s in enumerate(self.dataset.all_samples):
            gloss = s["gloss"]
            target = s["target_word"]
            self.gloss_to_indices[gloss].append(idx)
            self.target_to_glosses[target].add(gloss)
            self.gloss_to_target[gloss] = target

        # total samples
        self.num_samples = len(self.dataset)
        # number of batches per epoch (ceil to cover all)
        self._num_batches = ceil(self.num_samples / self.batch_size)

    def __len__(self):
        return self._num_batches

    def __iter__(self):
        rnd = random.Random(self.seed)
        # create mutable deques of indices per gloss (so we pop used indices)
        gloss_queues = {g: deque(idxs) for g, idxs in self.gloss_to_indices.items()}
        # set of glosses that still have elements
        available_glosses = set(g for g, q in gloss_queues.items() if q)
        # mapping target -> list of glosses currently available for that target
        target_available = {t: set(gset) & available_glosses for t, gset in self.target_to_glosses.items()}

        # For convenience, make list of targets
        targets = list(self.target_to_glosses.keys())

        batches_yielded = 0
        # Continue until all indices consumed
        while any(len(q) > 0 for q in gloss_queues.values()):
            batch = []
            used_glosses = set()   # track glosses used in this batch to avoid duplicates
            # Step 1: if prefer_same_target, try to select a target with >=2 available glosses
            if self.prefer_same_target:
                # find candidate targets that still have at least 2 different glosses available
                candidates = [t for t, gset in target_available.items() if len(gset) >= 2]
                rnd.shuffle(candidates)
                for t in candidates:
                    # pick up to min(remaining capacity in batch, number of glosses available for this target)
                    avail_glosses = list(target_available[t] - used_glosses)
                    if not avail_glosses:
                        continue
                    rnd.shuffle(avail_glosses)
                    # Add different glosses from same target (one index per gloss)
                    for g in avail_glosses:
                        if len(batch) >= self.batch_size:
                            break
                        # pop one index from gloss queue
                        if gloss_queues[g]:
                            batch.append(gloss_queues[g].popleft())
                            used_glosses.add(g)
                            # update availability structures below after loop
                    if len(batch) >= self.batch_size:
                        break

            # Step 2: fill remaining slots by sampling glosses (avoid used_glosses)
            remaining_slots = self.batch_size - len(batch)
            if remaining_slots > 0:
                # create list of glosses available excluding used_glosses
                candidate_glosses = [g for g in available_glosses if g not in used_glosses]
                rnd.shuffle(candidate_glosses)
                for g in candidate_glosses:
                    if remaining_slots <= 0:
                        break
                    if not gloss_queues[g]:
                        continue
                    batch.append(gloss_queues[g].popleft())
                    used_glosses.add(g)
                    remaining_slots -= 1

            # Step 3: If still slots left (rare — e.g., only one gloss remains but with many indices),
            # we must relax the constraint and allow same gloss multiple times in the batch.
            if len(batch) < self.batch_size:
                # pick any glosses with remaining elements (including those in used_glosses)
                leftover_glosses = [g for g, q in gloss_queues.items() if q]
                if leftover_glosses:
                    rnd.shuffle(leftover_glosses)
                    for g in leftover_glosses:
                        if len(batch) >= self.batch_size:
                            break
                        if gloss_queues[g]:
                            batch.append(gloss_queues[g].popleft())

            # After selecting batch items, update available_glosses and target_available
            # Remove glosses with empty queues from available_glosses and update targets
            to_remove = [g for g, q in gloss_queues.items() if not q and g in available_glosses]
            for g in to_remove:
                available_glosses.discard(g)
                t = self.gloss_to_target.get(g)
                if t is not None and t in target_available and g in target_available[t]:
                    target_available[t].discard(g)

            # yield batch (may be shorter than batch_size at very end)
            if batch:
                yield batch
                batches_yielded += 1
            else:
                # nothing left; break to avoid infinite loop
                break

            # small safety: advance seed slightly so shuffle differs per epoch if desired
            # (we do not change self.seed permanently)
            self.seed += 1

        # ensure we yield exactly all indices across batches (no repeats)
        # (consumer can verify by summing len of yielded batches == self.num_samples)

