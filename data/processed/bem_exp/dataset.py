import random
import re
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import TargetWordMaskExtractor  # Updated import
from utils.process_data import text_normalize
from collections import defaultdict
from collections import defaultdict
import random

class BEMDataset(Dataset):
    def __init__(self, samples, tokenizer, train_gloss_size=5, val_mode=False):  # Giảm từ 256 xuống 5
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        self.mask_extractor = TargetWordMaskExtractor(tokenizer)
        self.val_mode = val_mode
        self.train_gloss_size = train_gloss_size

        self.targetword2glosses = defaultdict(list)
        self.gloss_to_id = {}
        self.sense_weights = {}  # Thêm sense weights như code gốc
        
        word_set = set()
        self.all_samples = []
        gloss_set = set()

        # Count glosses per target word for weighting
        gloss_counts = defaultdict(lambda: defaultdict(int))
        
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
            
            self.gloss_to_id[gloss] = gloss_id
            gloss_counts[target][gloss] += 1
            
            word_set.add(int(word_id))
            gloss_set.add(gloss)

        # Compute sense weights như code gốc
        for target_word in self.targetword2glosses:
            glosses = self.targetword2glosses[target_word]
            counts = [gloss_counts[target_word][gloss] for gloss in glosses]
            total_count = sum(counts)
            # Inverse frequency weighting
            weights = [total_count / count for count in counts]
            self.sense_weights[target_word] = torch.FloatTensor(weights)

        sorted_wids = sorted(word_set)
        self.global_word_to_label = {wid: i for i, wid in enumerate(sorted_wids)}
        self.global_gloss_pool = list(gloss_set)

        # Pre-compute target masks
        self.target_masks = []
        print("Computing target masks...")
        for s in tqdm(self.all_samples, desc="Computing target masks", ascii=True):
            try:
                input_ids, attention_mask, target_mask = self.mask_extractor.extract_target_mask(
                    s["sentence"], s["target_word"], case_sensitive=False
                )
                self.target_masks.append(target_mask)
            except Exception as e:
                print(f"Warning: Failed to extract mask for '{s['target_word']}' in '{s['sentence']}': {e}")
                self.target_masks.append(torch.zeros(1, dtype=torch.long))

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
            "target_mask": self.target_masks[idx],
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
        target_masks = [b["target_mask"] for b in batch]
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

        # Process target masks
        batch_target_masks = self._align_target_masks_with_tokens(
            sentences, target_words, c_toks["input_ids"], target_masks
        )

        # Handle candidate glosses - FIXED STRATEGY
        final_candidates = []
        gold_indices = []
        sense_weights_batch = []
        
        for i, b in enumerate(batch):
            gold_gloss = b["gold_gloss"]
            target_word = b["target_word"]
            candidates = b["candidate_glosses"].copy()
            
            if not self.val_mode:
                # Training: FIXED sampling strategy
                if len(candidates) > self.train_gloss_size:
                    # Always keep gold at position 0
                    others = [c for c in candidates if c != gold_gloss]
                    sampled_others = random.sample(others, self.train_gloss_size - 1)
                    candidates = [gold_gloss] + sampled_others
                elif len(candidates) < self.train_gloss_size:
                    # Add random glosses from global pool
                    needed = self.train_gloss_size - len(candidates)
                    available = [g for g in self.global_gloss_pool if g not in candidates]
                    if len(available) >= needed:
                        extra = random.sample(available, needed)
                        # Gold vẫn ở đầu
                        non_gold = [c for c in candidates if c != gold_gloss]
                        candidates = [gold_gloss] + non_gold + extra
                
                # KHÔNG shuffle - gold luôn ở position 0
                gold_idx = 0
            else:
                # Validation: không shuffle
                if gold_gloss != candidates[0]:
                    candidates.remove(gold_gloss)
                    candidates.insert(0, gold_gloss)
                gold_idx = 0
            
            final_candidates.append(candidates)
            gold_indices.append(gold_idx)
            
            # Get sense weights for this target word
            if target_word in self.sense_weights:
                weights = self.sense_weights[target_word]
                # Align weights with current candidates
                candidate_weights = []
                for cand in candidates:
                    try:
                        cand_idx = self.targetword2glosses[target_word].index(cand)
                        candidate_weights.append(weights[cand_idx].item())
                    except (ValueError, IndexError):
                        candidate_weights.append(1.0)  # Default weight
                sense_weights_batch.append(torch.FloatTensor(candidate_weights))
            else:
                sense_weights_batch.append(torch.ones(len(candidates)))

        return {
            "context_input_ids": c_toks["input_ids"],
            "context_attn_mask": c_toks["attention_mask"],
            "target_masks": batch_target_masks,
            "synset_ids": synset_ids,
            "word_id": word_id,
            "gold_glosses": gold_glosses,
            "target_words": target_words,
            "candidate_glosses": final_candidates,
            "gloss_ids": gloss_ids,
            "gold_indices": torch.tensor(gold_indices, dtype=torch.long),
            "sense_weights": sense_weights_batch  # Thêm weights
        }

    def _align_target_masks_with_tokens(self, sentences, target_words, tokenized_input_ids, pre_computed_masks):
        batch_size, seq_len = tokenized_input_ids.shape
        aligned_masks = torch.zeros((batch_size, seq_len), dtype=torch.long)
        
        for i, (sentence, target_word) in enumerate(zip(sentences, target_words)):
            try:
                _, _, target_mask = self.mask_extractor.extract_target_mask(
                    sentence, target_word, case_sensitive=False
                )
                mask_len = min(len(target_mask), seq_len)
                aligned_masks[i, :mask_len] = target_mask[:mask_len]
            except Exception as e:
                print(f"Warning: Failed to align mask for '{target_word}' in sentence {i}: {e}")
                continue
        
        return aligned_masks
