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
            "target_words": target_words,
            "candidate_glosses": candidate_glosses,
            "gloss_id": gloss_id
        }

class PolyBERTtDatasetV3(Dataset):
    def __init__(self, samples, tokenizer):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)

        self.all_samples = []
        self.global_gloss_pool = []
        self.word2gloss = defaultdict(list)

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
        return {
            "sentence": s["sentence"],
            "target_word": s["target_word"],
            "target_span": self.span_indices[idx],
            "gloss": s["gloss"],
            "gloss_id": s["gloss_id"],
            "word_id": s["word_id"],
            "synset_id": s["synset_id"]
        }


class ContrastiveBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, num_negatives=16, drop_last=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_negatives = num_negatives
        self.drop_last = drop_last
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        random.shuffle(self.indices)
        batch = []
        for idx in self.indices:
            batch.append(idx)
            if len(batch) == self.batch_size:
                # build candidate glosses cho batch
                batch_candidates = []
                for i in batch:
                    gold = self.dataset[i]["gloss"]
                    # lấy negatives từ global pool
                    negs = random.sample(
                        [g for g in self.dataset.global_gloss_pool if g != gold],
                        k=self.num_negatives
                    )
                    cands = [gold] + negs
                    batch_candidates.append(cands)

                yield batch, batch_candidates
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.indices) // self.batch_size
        else:
            return math.ceil(len(self.indices) / self.batch_size)
