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
        synset_set = set()
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
                "word_id":word_id
            })
            
            if self.val_mode:
                if gloss not in self.targetword2glosses[target]:
                    self.targetword2glosses[target].append(gloss)

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
        item = {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "synset_id": s["synset_id"],
            "word_id": s.get("word_id"),
            "target_word": s["target_word"],
            "gloss" : s["gloss"]
        }
        if self.val_mode:
            # candidate glosses = all glosses observed for this target_word
            item["candidate_glosses"] = list(self.targetword2glosses[s["target_word"]])
        else:
            item["gloss"] = s["gloss"]

        return item

    def collate_fn(self, batch):
        if self.val_mode:
            contexts = []
            glosses = []
            labels = []
            target_words = []
            word_ids = []
            spans = []

            for b in batch:
                cand_glosses = b["candidate_glosses"]
                contexts.extend([b["sentence"]] * len(cand_glosses))
                glosses.extend(cand_glosses)
                spans.extend([b["target_span"]] * len(cand_glosses))
                target_words.extend([b["target_word"]] * len(cand_glosses))
                word_ids.extend([b["word_id"]] * len(cand_glosses))

                labels.extend([
                    1 if g == b["gloss"] else 0
                    for g in cand_glosses
                ])

            c_toks = self.tokenizer(
                contexts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )
            g_toks = self.tokenizer(
                glosses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            )

            return {
                "context_input_ids": c_toks["input_ids"],
                "context_attn_mask": c_toks["attention_mask"],
                "target_spans": torch.tensor(spans, dtype=torch.long),
                "synset_labels": torch.tensor(labels, dtype=torch.long),  
                "word_id": torch.tensor(word_ids, dtype=torch.long),
                "target_words": target_words,
                "gloss_input_ids": g_toks["input_ids"],
                "gloss_attn_mask": g_toks["attention_mask"],
            }

        sentences = [b["sentence"] for b in batch]
        target_words = [b["target_word"] for b in batch]
        spans     = [b["target_span"] for b in batch]
        labels    = torch.tensor([b["synset_id"] for b in batch], dtype=torch.long)
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
    Batch sampler for contrastive learning with priorities:
    1. Samples with same target word (lemma) but different word_id/synset_id are grouped together
    2. No samples with identical (lemma, word_id, synset_id) in the same batch
    """
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Build data structures
        self.lemma2keys = defaultdict(set)  # lemma -> set of (word_id, synset_id)
        self.key2indices = defaultdict(list)  # (lemma, word_id, synset_id) -> list of indices
        
        for idx, item in enumerate(dataset):
            lemma = item['target_word']
            word_id = item['word_id']
            synset_id = item['synset_ids']
            
            key = (lemma, word_id, synset_id)
            self.lemma2keys[lemma].add(key)
            self.key2indices[key].append(idx)
        
        # Convert to persistent structures
        self.lemmas = list(self.lemma2keys.keys())
        self.lemma_keys = {lemma: list(keys) for lemma, keys in self.lemma2keys.items()}
    
    def __iter__(self):
        # Create mutable copies for this epoch
        key2indices = {key: idxs[:] for key, idxs in self.key2indices.items()}
        lemma_keys = {lemma: keys[:] for lemma, keys in self.lemma_keys.items()}
        
        # Prepare shuffled list of lemmas
        lemmas = self.lemmas[:]
        random.shuffle(lemmas)
        
        # Process each lemma
        for lemma in lemmas:
            # Get available keys for this lemma
            available_keys = [
                key for key in lemma_keys[lemma] 
                if key in key2indices and key2indices[key]
            ]
            random.shuffle(available_keys)
            
            while available_keys:
                batch = []
                selected_keys = []
                
                # Try to fill batch with current lemma
                for key in available_keys:
                    if len(batch) >= self.batch_size:
                        break
                    
                    # Add sample from this key
                    idx = key2indices[key].pop(0)
                    batch.append(idx)
                    selected_keys.append(key)
                    
                    # Remove key if no more samples
                    if not key2indices[key]:
                        del key2indices[key]
                
                # Remove selected keys from processing list
                available_keys = [k for k in available_keys if k not in selected_keys]
                
                # Yield batch if we have any samples
                if batch:
                    yield batch
    
    def __len__(self):
        # Estimate based on total samples
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
