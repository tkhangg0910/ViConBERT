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


class ContrastiveBatchSampler(Sampler):
    """
    Sampler for PolyBERT contrastive training:
    - Avoid multiple identical gloss in batch
    - Encourage same target_word but different gloss
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

        # Map target_word -> list of sample indices
        self.word2indices = defaultdict(list)
        # Map gloss -> list of sample indices (to avoid duplicates in batch)
        self.gloss2indices = defaultdict(list)

        for idx, item in enumerate(dataset):
            self.word2indices[item["target_word"]].append(idx)
            self.gloss2indices[item["gloss"]].append(idx)

        self.words = list(self.word2indices.keys())

    def __iter__(self):
        batch = []
        used_glosses = set()

        words = self.words[:]
        random.shuffle(words)

        for word in words:
            candidates = self.word2indices[word][:]
            random.shuffle(candidates)

            for idx in candidates:
                gloss = self.dataset[idx]["gloss"]
                if gloss in used_glosses:
                    continue  # skip duplicate gloss in current batch
                batch.append(idx)
                used_glosses.add(gloss)

                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                    used_glosses = set()

        # yield leftover batch
        if batch:
            yield batch

    def __len__(self):
        # approximate number of batches
        return len(self.dataset) // self.batch_size