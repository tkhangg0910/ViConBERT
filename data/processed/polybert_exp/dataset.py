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
        if val_mode == False:
            gloss_set = set()
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
            if val_mode == False:
              gloss_set.add(gloss)
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
        if val_mode == False:
            self.gloss_list = list(gloss_set)
            self.g_tokes = self.tokenizer(
                self.gloss_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True,
                return_offsets_mapping=True
            )
            self.gloss_input_ids= self.g_tokes["input_ids"]
            self.gloss_attn_mask=self.g_tokes["attention_mask"]
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
        # g_tokes = self.tokenizer(
        #     glosses,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=256,
        #     return_attention_mask=True,
        #     return_offsets_mapping=True
        # )
        gold_glosses_idx = torch.tensor([
            self.gloss_list.index(g) for g in glosses
        ], dtype=torch.long)
        return {
            "context_input_ids": c_toks["input_ids"],
            "context_attn_mask": c_toks["attention_mask"],
            "target_spans": torch.tensor(spans, dtype=torch.long) if spans else None,
            "synset_ids": labels,
            "word_id":word_id,
            "gold_glosses_idx": gold_glosses_idx,
            "target_words":target_words,
            # "gloss_input_ids": self.g_tokes["input_ids"],
            # "gloss_attn_mask": self.g_tokes["attention_mask"],
            "gloss_id":gloss_id
        }

class PolyBERTtDataseV2(Dataset):
    def __init__(self, samples, tokenizer, train_gloss_size=256, val_mode=False):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)
        self.val_mode = val_mode
        self.train_gloss_size = train_gloss_size

        self.targetword2glosses = defaultdict(list)

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

            word_set.add(int(word_id))
            gloss_set.add(gloss)

        sorted_wids = sorted(word_set)
        self.global_word_to_label = {wid: i for i, wid in enumerate(sorted_wids)}

        # global gloss pool for random sampling
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
        item = {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "word_id": label,
            "target_word": s["target_word"],
            "gloss": s["gloss"],
            "synset_ids": sid,
            "gloss_id": gid
        }

        gold_gloss = s["gloss"]
        item["candidate_glosses"] = list(set(self.targetword2glosses[s["target_word"]] + [gold_gloss]))

        return item

    def collate_fn(self, batch):
        sentences = [b["sentence"] for b in batch]
        gold_glosses = [b["gloss"] for b in batch]

        spans = [b["target_span"] for b in batch]
        synset_ids = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
        target_words = [b["target_word"] for b in batch]
        gloss_id = torch.tensor([b["gloss_id"] for b in batch], dtype=torch.long)

        c_toks = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True
        )

        if self.val_mode:
            candidate_glosses = [b["candidate_glosses"] for b in batch]
        else:
            candidate_glosses = []
            for b in batch:
                gold_gloss = b["gloss"]
                cands = set(b["candidate_glosses"])  # unique
                # đảm bảo gold gloss luôn có mặt
                cands.add(gold_gloss)
                needed = self.train_gloss_size - len(cands)
                if needed > 0:
                    extra = random.sample(
                        [g for g in self.global_gloss_pool if g not in cands],
                        k=min(needed, len(self.global_gloss_pool) - len(cands))
                    )
                    cands.update(extra)
                # nếu cands > train_gloss_size thì random truncate
                if len(cands) > self.train_gloss_size:
                    cands = set(random.sample(list(cands), self.train_gloss_size))
                candidate_glosses.append(list(cands))


        return {
            "context_input_ids": c_toks["input_ids"],
            "context_attn_mask": c_toks["attention_mask"],
            "target_spans": torch.tensor(spans, dtype=torch.long),
            "synset_ids": synset_ids,
            "word_id": word_id,
            "gold_glosses": gold_glosses ,
            "target_words": target_words,
            "candidate_glosses": candidate_glosses,
            "gloss_id": gloss_id
        }
        
class PolyBERTtDatasetV3(Dataset):
    def __init__(self, samples, tokenizer, val_mode=False):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)

        self.all_samples = []
        self.global_gloss_pool = []
        self.word2gloss = defaultdict(list)
        self.val_mode = val_mode

        for sample in samples:
            sent = text_normalize(sample["sentence"])
            item = {
                "sentence": sent,
                "target_word": sample["target_word"],
                "synset_id": sample["synset_id"],
                "gloss": sample["gloss"],
                "gloss_id": sample["gloss_id"],
                "word_id": int(sample["word_id"])
            }
            self.all_samples.append(item)
            self.word2gloss[item["target_word"]].append(item["gloss"])
            self.global_gloss_pool.append(item["gloss"])

        self.global_gloss_pool = list(set(self.global_gloss_pool))

        # span indices
        self.span_indices = []
        for s in tqdm(self.all_samples, desc="Computing spans", ascii=True):
            idxs = self.span_extractor.get_span_indices(
                s["sentence"], s["target_word"]
            )
            self.span_indices.append(idxs or (0,0))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s = self.all_samples[idx]
        item = {
            "sentence": s["sentence"],
            "target_word": s["target_word"],
            "target_spans": self.span_indices[idx],
            "gloss": s["gloss"],
            "gloss_id": s["gloss_id"],
            "word_id": s["word_id"],
            "synset_id": s["synset_id"]
        }
        if self.val_mode:
            item["candidate_glosses"] = list(set(self.word2gloss[s["target_word"]]))
        return item

    def collate_fn(self, batch):
        if self.val_mode:
            sentences = [b["sentence"] for b in batch]          # B
            spans     = [b["target_spans"] for b in batch]       # B x (2)
            synset_ids = torch.tensor([b["synset_id"] for b in batch], dtype=torch.long)
            word_id = torch.tensor([(b["word_id"]) for b in batch], dtype=torch.long)
            target_words = [b["target_word"] for b in batch]
            gold_glosses = [b["gloss"] for b in batch]
            gloss_id = torch.tensor([(b["gloss_id"]) for b in batch], dtype=torch.long)
            candidate_glosses = [b["candidate_glosses"] for b in batch]
        
            return {
                "sentence":sentences,
                "target_spans": torch.tensor(spans, dtype=torch.long),  # [B,2]
                "synset_ids": synset_ids,
                "word_id": word_id,
                "target_words": target_words,
                "gloss": gold_glosses,                
                "candidate_glosses": candidate_glosses,
                "gloss_id":gloss_id 
            }

        sentences = [b["sentence"] for b in batch]
        target_words = [b["target_word"] for b in batch]
        spans     = [b["target_spans"] for b in batch]
        labels    = torch.tensor([b["synset_id"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
        glosses = [b["gloss"] for b in batch]
        gloss_id = torch.tensor([(b["gloss_id"]) for b in batch], dtype=torch.long)
        return {
            "sentence":sentences,
            "target_spans": torch.tensor(spans, dtype=torch.long) if spans else None,
            "synset_ids": labels,
            "word_id":word_id,
            "gloss": glosses,
            "target_words":target_words,
            "gloss_id":gloss_id
        }
class ContrastiveBatchSampler(Sampler):
    """
    Batch sampler for contrastive WSD batches.

    For batch_size = N (must be even):
      - choose N/2 anchor indices with distinct glosses
      - for each anchor, choose one "candidate" index:
          - prefer an index whose gloss is in dataset.word2gloss[anchor_target_word]
            (and != anchor.gloss) if such sample exists and gloss not already used
          - otherwise fallback to a random sample whose gloss is not used yet
      - ensure *no gloss duplicates* inside the returned batch

    Args:
      dataset: instance of PolyBERTtDatasetV3 (or similar)
      batch_size: even integer
      shuffle: whether to shuffle anchors each epoch
      max_tries: how many attempts to find valid batch before restarting attempt
    """
    def __init__(self, dataset, batch_size, shuffle=True, max_tries=100):
        assert batch_size % 2 == 0, "batch_size must be even"
        self.dataset = dataset
        self.bs = batch_size
        self.half = batch_size // 2
        self.shuffle = shuffle
        self.max_tries = max_tries

        # build gloss -> indices mapping for quick lookup
        self.gloss_to_indices = defaultdict(list)
        for idx, item in enumerate(self.dataset.all_samples):
            gloss = item["gloss"]
            self.gloss_to_indices[gloss].append(idx)

        # list of all dataset indices
        self.indices = list(range(len(self.dataset)))

        # list of all unique glosses (strings)
        self.unique_glosses = list(self.gloss_to_indices.keys())

        # precompute index -> gloss mapping for speed
        self.idx2gloss = {idx: self.dataset.all_samples[idx]["gloss"] for idx in self.indices}
        # maps index -> target_word (used to get candidate glosses)
        self.idx2word = {idx: self.dataset.all_samples[idx]["target_word"] for idx in self.indices}

    def __len__(self):
        # number of batches per epoch
        return len(self.dataset) // self.bs

    def __iter__(self):
        # anchors candidate order
        anchors_pool = self.indices.copy()
        if self.shuffle:
            random.shuffle(anchors_pool)

        # We'll iterate until we produce floor(N_samples / bs) batches
        produced = 0
        i = 0
        total_batches = len(self)

        while produced < total_batches:
            tries = 0
            batch = None

            while tries < self.max_tries:
                tries += 1
                # pick N/2 anchors ensuring distinct glosses
                anchors = []
                used_glosses = set()

                # try scanning anchors_pool from current pos to pick half anchors
                j = i
                while len(anchors) < self.half and j < len(anchors_pool):
                    cand_idx = anchors_pool[j]
                    cand_gloss = self.idx2gloss[cand_idx]
                    if cand_gloss not in used_glosses:
                        anchors.append(cand_idx)
                        used_glosses.add(cand_gloss)
                    j += 1

                # if not enough anchors available in the tail, reshuffle and restart selection
                if len(anchors) < self.half:
                    # reshuffle anchors and restart picking
                    if self.shuffle:
                        random.shuffle(anchors_pool)
                    i = 0
                    continue

                # Now for each anchor pick a candidate index whose gloss is from anchor's word's gloss pool
                candidates = []
                success = True
                for a_idx in anchors:
                    a_word = self.idx2word[a_idx]
                    a_gloss = self.idx2gloss[a_idx]

                    # candidate gloss strings available for this word
                    word_glosses = list(self.dataset.word2gloss.get(a_word, []))
                    # remove the anchor's gloss itself
                    word_glosses = [g for g in word_glosses if g != a_gloss]

                    # shuffle to randomize selection
                    random.shuffle(word_glosses)

                    found = False
                    # try to pick a gloss from word_glosses that's not yet used in batch and has a sample
                    for g in word_glosses:
                        if g in used_glosses:
                            continue
                        idx_list = self.gloss_to_indices.get(g, [])
                        if not idx_list:
                            continue
                        # pick random sample index with gloss g
                        cand_sample = random.choice(idx_list)
                        # ensure it's not the same as anchor
                        if cand_sample == a_idx:
                            # try another from list
                            if len(idx_list) == 1:
                                continue
                            else:
                                # pick until different or skip
                                tries_inner = 0
                                while cand_sample == a_idx and tries_inner < 5:
                                    cand_sample = random.choice(idx_list)
                                    tries_inner += 1
                                if cand_sample == a_idx:
                                    continue
                        # accept
                        candidates.append(cand_sample)
                        used_glosses.add(g)
                        found = True
                        break

                    if not found:
                        # fallback: pick any random index whose gloss is not used yet
                        # sample from global_gloss_pool list of glosses
                        fallback_found = False
                        random.shuffle(self.unique_glosses)
                        for g in self.unique_glosses:
                            if g in used_glosses:
                                continue
                            idx_list = self.gloss_to_indices.get(g, [])
                            if not idx_list:
                                continue
                            cand_sample = random.choice(idx_list)
                            if cand_sample == a_idx:
                                continue
                            candidates.append(cand_sample)
                            used_glosses.add(g)
                            fallback_found = True
                            break

                        if not fallback_found:
                            success = False
                            break

                if not success:
                    # try again
                    if self.shuffle:
                        random.shuffle(anchors_pool)
                    i = 0
                    continue

                # combine anchors + candidates => final batch indices
                batch = []
                for a, c in zip(anchors, candidates):
                    batch.append(a)
                    batch.append(c)

                # final check: ensure batch length == bs and gloss uniqueness
                if len(batch) == self.bs and len(used_glosses) == self.bs:
                    # optionally shuffle the order inside batch to avoid anchor-candidate pairing fixed positions
                    random.shuffle(batch)
                    # advance i so we don't repeatedly pick same anchors (simple stride)
                    i = j
                    break
                else:
                    # try again
                    if self.shuffle:
                        random.shuffle(anchors_pool)
                    i = 0
                    continue

            if batch is None:
                # If after many tries failed, fallback to random batch with unique gloss constraint best-effort
                used_glosses = set()
                batch = []
                random.shuffle(self.indices)
                for idx in self.indices:
                    g = self.idx2gloss[idx]
                    if g in used_glosses:
                        continue
                    batch.append(idx)
                    used_glosses.add(g)
                    if len(batch) >= self.bs:
                        break
                if len(batch) < self.bs:
                    # pad with random indices (less safe)
                    while len(batch) < self.bs:
                        batch.append(random.choice(self.indices))

            produced += 1
            yield batch

    def set_epoch(self, epoch):
        # optional: reseed/reshuffle per epoch externally if needed
        if self.shuffle:
            random.shuffle(self.indices)
