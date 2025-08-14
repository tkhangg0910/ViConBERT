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
    Smart sampler for contrastive learning that:
    1. Avoids duplicate glosses in the same batch (for better negative sampling)
    2. Encourages same target_word + different gloss in batch (for positive pairs)
    3. Guarantees consistent number of batches
    """
    def __init__(self, dataset, batch_size, drop_last=False, 
                 min_positives_per_batch=2, max_attempts=1000):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.min_positives_per_batch = min_positives_per_batch
        self.max_attempts = max_attempts
        
        # Build indexes
        self._build_indexes()
        
        # Print statistics
        self._print_stats()
    
    def _build_indexes(self):
        """Build efficient lookup indexes"""
        self.word2indices = defaultdict(list)     # word -> [sample_indices]
        self.gloss2indices = defaultdict(list)    # gloss -> [sample_indices] 
        self.word_gloss2indices = defaultdict(list)  # (word,gloss) -> [sample_indices]
        
        for idx, item in enumerate(self.dataset):
            word = item["target_word"]
            gloss = item["gloss"]
            
            self.word2indices[word].append(idx)
            self.gloss2indices[gloss].append(idx)
            self.word_gloss2indices[(word, gloss)].append(idx)
        
        # Find words with multiple glosses (good for positive pairs)
        self.multi_gloss_words = {}
        for word, indices in self.word2indices.items():
            glosses = set(self.dataset[idx]["gloss"] for idx in indices)
            if len(glosses) >= 2:  # Word has multiple different glosses
                self.multi_gloss_words[word] = list(glosses)
    
    def _print_stats(self):
        """Print dataset statistics for debugging"""
        total_words = len(self.word2indices)
        total_glosses = len(self.gloss2indices)
        multi_gloss_words = len(self.multi_gloss_words)
        
        avg_glosses_per_word = sum(len(set(self.dataset[idx]["gloss"] 
                                          for idx in indices))
                                  for indices in self.word2indices.values()) / total_words
        
        print(f"=== SmartContrastiveBatchSampler Stats ===")
        print(f"Total samples: {len(self.dataset)}")
        print(f"Unique words: {total_words}")
        print(f"Unique glosses: {total_glosses}")
        print(f"Words with multiple glosses: {multi_gloss_words}")
        print(f"Average glosses per word: {avg_glosses_per_word:.2f}")
        print(f"Expected batches: {len(self)}")
    
    def __iter__(self):
        """Generate batches with smart contrastive sampling"""
        all_indices = list(range(len(self.dataset)))
        random.shuffle(all_indices)
        
        used_indices = set()
        
        while len(used_indices) < len(self.dataset):
            batch = self._create_smart_batch(all_indices, used_indices)
            
            if not batch:  # No more valid batches can be created
                break
                
            if len(batch) >= self.min_positives_per_batch or not self.drop_last:
                yield batch
                used_indices.update(batch)
            else:
                break
        
        # Handle remaining samples
        remaining = [idx for idx in all_indices if idx not in used_indices]
        if remaining and not self.drop_last:
            # Create final batch with remaining samples
            final_batch = self._create_final_batch(remaining)
            if final_batch:
                yield final_batch
    
    def _create_smart_batch(self, all_indices, used_indices):
        """Create a batch optimized for contrastive learning"""
        batch = []
        used_glosses = set()
        used_words = set()
        
        # Available indices
        available_indices = [idx for idx in all_indices if idx not in used_indices]
        if not available_indices:
            return batch
        
        # Strategy 1: Start with a word that has multiple glosses
        starter_word = self._find_multi_gloss_starter(available_indices, used_indices)
        if starter_word:
            # Add samples from this word with different glosses
            word_samples = self._add_word_samples(starter_word, available_indices, 
                                                used_indices, batch, used_glosses)
            batch.extend(word_samples)
            used_words.add(starter_word)
            
            # Update used glosses
            for idx in word_samples:
                used_glosses.add(self.dataset[idx]["gloss"])
        
        # Strategy 2: Fill remaining slots avoiding gloss duplicates
        attempts = 0
        while len(batch) < self.batch_size and attempts < self.max_attempts:
            attempts += 1
            
            # Find candidates that don't duplicate glosses
            candidates = []
            for idx in available_indices:
                if idx in batch or idx in used_indices:
                    continue
                    
                gloss = self.dataset[idx]["gloss"]
                word = self.dataset[idx]["target_word"]
                
                if gloss not in used_glosses:
                    # Prioritize words not yet in batch (for diversity)
                    priority = 2 if word not in used_words else 1
                    candidates.append((priority, idx))
            
            if not candidates:
                break
                
            # Sort by priority and pick best candidate
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, chosen_idx = candidates[0]
            
            batch.append(chosen_idx)
            used_glosses.add(self.dataset[chosen_idx]["gloss"])
            used_words.add(self.dataset[chosen_idx]["target_word"])
        
        return batch
    
    def _find_multi_gloss_starter(self, available_indices, used_indices):
        """Find a good starting word with multiple available glosses"""
        word_available_glosses = defaultdict(set)
        
        # Count available glosses per word
        for idx in available_indices:
            if idx in used_indices:
                continue
            word = self.dataset[idx]["target_word"]
            gloss = self.dataset[idx]["gloss"]
            word_available_glosses[word].add(gloss)
        
        # Find word with most available glosses
        best_word = None
        max_glosses = 0
        
        for word, glosses in word_available_glosses.items():
            if len(glosses) > max_glosses:
                max_glosses = len(glosses)
                best_word = word
        
        return best_word if max_glosses >= 2 else None
    
    def _add_word_samples(self, word, available_indices, used_indices, batch, used_glosses):
        """Add samples from a word with different glosses"""
        word_samples = []
        word_glosses = set()
        
        # Collect available samples for this word
        candidates = []
        for idx in available_indices:
            if (idx not in used_indices and idx not in batch and 
                self.dataset[idx]["target_word"] == word):
                gloss = self.dataset[idx]["gloss"]
                if gloss not in used_glosses and gloss not in word_glosses:
                    candidates.append(idx)
                    word_glosses.add(gloss)
        
        # Add up to min_positives_per_batch samples from this word
        random.shuffle(candidates)
        max_samples = min(len(candidates), self.min_positives_per_batch, 
                         self.batch_size - len(batch))
        
        word_samples = candidates[:max_samples]
        return word_samples
    
    def _create_final_batch(self, remaining_indices):
        """Create final batch from remaining samples"""
        if not remaining_indices:
            return []
            
        # Try to avoid gloss duplicates even in final batch
        batch = []
        used_glosses = set()
        
        # First pass: add samples with unique glosses
        for idx in remaining_indices:
            gloss = self.dataset[idx]["gloss"]
            if gloss not in used_glosses and len(batch) < self.batch_size:
                batch.append(idx)
                used_glosses.add(gloss)
        
        # Second pass: fill remaining slots if needed
        for idx in remaining_indices:
            if idx not in batch and len(batch) < self.batch_size:
                batch.append(idx)
        
        return batch
    
    def __len__(self):
        """Return expected number of batches"""
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

