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
        self.gloss_set = list(gloss_set)
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
            self.g_tokes = self.tokenizer(
                gloss_set,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True,
                return_offsets_mapping=True
            )
            self.gloss_input_ids= self.g_tokes["input_ids"],
            self.gloss_attn_mask=self.g_tokes["attention_mask"],
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
            self.gloss_set.index(g) for g in glosses
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


class ContrastiveBatchSampler(Sampler):
    """
    Batch sampler that:
    - tries to maximize distinct gloss_id inside each batch
    - optionally adds pairs of samples that share the same target_word but come
      from different gloss_id (hard negatives)
    Use with DataLoader(..., batch_sampler=ContrastiveBatchSampler(...))
    """
    def __init__(self, dataset, batch_size, same_word_pair_prob=0.3, drop_last=True, shuffle=True, seed=None):
        """
        dataset: your PolyBERTtDataset instance (expects dataset.all_samples with 'gloss_id' and 'target_word')
        batch_size: int
        same_word_pair_prob: fraction of batch size to allocate to same-target-word pairs (range 0..1).
                             e.g. 0.3 -> ~30% of batch positions used in same-word pairs (pairs consume 2 positions)
        drop_last: whether to drop last incomplete batch
        shuffle: shuffle samples within each gloss list at start of epoch
        """
        assert batch_size >= 2, "batch_size should be >= 2"
        self.dataset = dataset
        self.batch_size = batch_size
        self.same_word_pair_prob = float(same_word_pair_prob)
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed

        # Build mapping gloss_id -> indices, and target_word -> set(gloss_id)
        gloss2idxs = defaultdict(list)
        word2glosses = defaultdict(set)

        for idx, s in enumerate(self.dataset.all_samples):
            gid = s["gloss_id"]
            w = s["target_word"]
            gloss2idxs[gid].append(idx)
            word2glosses[w].add(gid)

        # convert sets to lists
        self.gloss2idxs = {g: list(idxs) for g, idxs in gloss2idxs.items()}
        self.word2glosses = {w: list(gs) for w, gs in word2glosses.items()}

        # Precompute total samples
        self.total_samples = sum(len(v) for v in self.gloss2idxs.values())

        # reproducible randomness
        self._rnd = random.Random(seed)

    def __len__(self):
        # number of batches per epoch
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return math.ceil(self.total_samples / self.batch_size)

    def _prepare_epoch(self):
        # Make working copies of per-gloss indices and shuffle if needed
        self._work_gloss2idxs = {g: list(idxs) for g, idxs in self.gloss2idxs.items()}
        if self.shuffle:
            for idxs in self._work_gloss2idxs.values():
                self._rnd.shuffle(idxs)

        # Track glosses that currently have samples
        self._available_glosses = {g for g, idxs in self._work_gloss2idxs.items() if len(idxs) > 0}

        # precompute words that could possibly produce pairs (>=2 distinct glosses with >=1 sample)
        self._update_candidate_pair_words()

        # total remaining samples
        self._remaining = sum(len(idxs) for idxs in self._work_gloss2idxs.values())

    def _update_candidate_pair_words(self):
        self._candidate_pair_words = []
        for w, gloss_list in self.word2glosses.items():
            # count available glosses for this word with at least one sample
            avail = [g for g in gloss_list if g in self._available_glosses]
            if len(avail) >= 2:
                self._candidate_pair_words.append(w)

    def __iter__(self):
        self._prepare_epoch()
        batches_emitted = 0

        while self._remaining >= self.batch_size:
            batch = []
            used_gloss_in_batch = set()

            # how many positions to allocate to same-word pairs (each pair uses 2 positions)
            pair_positions = int(self.same_word_pair_prob * self.batch_size)
            num_pairs = pair_positions // 2  # full pairs only
            # safety: don't request more pairs than possible given remaining samples
            max_pairs_possible = 0
            # estimate max pairs by counting words that can still produce pairs
            for w in self._candidate_pair_words:
                avail_glosses = [g for g in self.word2glosses[w] if g in self._available_glosses]
                max_pairs_possible += len(avail_glosses) // 2  # loose bound
            num_pairs = min(num_pairs, max_pairs_possible)

            # create pairs
            for _ in range(num_pairs):
                if not self._candidate_pair_words:
                    break
                # pick a word that currently can produce a pair
                w = self._rnd.choice(self._candidate_pair_words)
                avail_glosses = [g for g in self.word2glosses[w] if g in self._available_glosses]
                if len(avail_glosses) < 2:
                    # update and continue
                    self._update_candidate_pair_words()
                    continue
                # choose two distinct gloss_ids
                g1, g2 = self._rnd.sample(avail_glosses, 2)

                # pop one index from each gloss
                i1 = self._work_gloss2idxs[g1].pop()
                i2 = self._work_gloss2idxs[g2].pop()
                batch.extend([i1, i2])
                used_gloss_in_batch.update([g1, g2])

                # update availability
                if not self._work_gloss2idxs[g1]:
                    self._available_glosses.discard(g1)
                if not self._work_gloss2idxs[g2]:
                    self._available_glosses.discard(g2)
                self._remaining -= 2

                # refresh candidate words list occasionally
                self._update_candidate_pair_words()

            # Fill remaining slots preferring distinct glosses
            slots_left = self.batch_size - len(batch)
            if slots_left > 0:
                # try to pick 'slots_left' distinct glosses different from used_gloss_in_batch
                candidates = list(self._available_glosses - used_gloss_in_batch)
                if len(candidates) >= slots_left:
                    chosen = self._rnd.sample(candidates, slots_left)
                else:
                    # not enough distinct glosses left -> take what we can, then allow duplicates
                    chosen = list(candidates)
                    needed = slots_left - len(chosen)
                    # make a flatten list of glosses with remaining samples (including those already used in batch)
                    refill_pool = [g for g in self._available_glosses]
                    # if still insufficient (rare), we will break early
                    if not refill_pool:
                        break
                    # sample with replacement if needed
                    for _ in range(needed):
                        chosen.append(self._rnd.choice(refill_pool))

                # pop one index from each chosen gloss
                for g in chosen:
                    if not self._work_gloss2idxs[g]:
                        continue
                    idx = self._work_gloss2idxs[g].pop()
                    batch.append(idx)
                    used_gloss_in_batch.add(g)
                    if not self._work_gloss2idxs[g]:
                        self._available_glosses.discard(g)
                    self._remaining -= 1

            # final verification: if batch has desired size, yield it
            if len(batch) == self.batch_size:
                batches_emitted += 1
                yield batch
            else:
                # not enough samples to complete batch
                break

        # optionally yield a smaller last batch if not drop_last
        if (not self.drop_last) and (self._remaining > 0):
            tail = []
            for g in list(self._available_glosses):
                while self._work_gloss2idxs[g] and len(tail) < self.batch_size:
                    tail.append(self._work_gloss2idxs[g].pop())
                    self._remaining -= 1
                if len(tail) >= self.batch_size:
                    break
            if tail:
                yield tail  # possibly smaller than batch_size

    def set_epoch_seed(self, seed):
        """Optional: set a new seed between epochs for reproducible shuffling."""
        self.seed = seed
        self._rnd = random.Random(seed)
