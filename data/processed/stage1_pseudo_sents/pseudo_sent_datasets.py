import random
from torch.utils.data import Dataset, WeightedRandomSampler,BatchSampler
from collections import defaultdict
from transformers import PreTrainedTokenizerFast
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize

class PseudoSents_Dataset(Dataset):
    def __init__(self, samples, tokenizer, num_synsets_per_batch=32, samples_per_synset=8):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(self.tokenizer)
        self.num_synsets_per_batch = num_synsets_per_batch
        self.samples_per_synset = samples_per_synset

        self.synset_groups = defaultdict(list)
        self.synset_to_group = {}
        self.all_samples = []

        for sample in tqdm(samples):
            # Normalize text
            normalized_sentence = text_normalize(sample["sentence"])
            normalized_target = sample["target_word"]
            
            # Get supersense group
            group = self._get_supersense_group(sample["supersense"])
            
            # Create new sample
            new_sample = {
                "sentence": normalized_sentence,
                "target_word": normalized_target,
                "synset_id": sample["synset_id"],
                "group": group
            }
            
            # Add to synset groups
            self.synset_groups[sample["synset_id"]].append(new_sample)
            self.synset_to_group[sample["synset_id"]] = group
            self.all_samples.append(new_sample)
            
        print("Precomputing span indices...")
        self.span_indices = []
        for sample in tqdm(self.all_samples):
            indices = self.span_extractor.get_span_indices(
                sample["sentence"], 
                sample["target_word"]
            )
            self.span_indices.append(indices if indices else (0, 0))
        
        # Create batch structure
        print("Organizing batches...")
        self.batches = []
        synset_ids = list(self.synset_groups.keys())
        
        for _ in range(len(synset_ids) // num_synsets_per_batch):
            # Select random synsets for this batch
            selected_synsets = random.sample(synset_ids, num_synsets_per_batch)
            
            batch_samples = []
            batch_synset_labels = []
            
            for i, synset_id in enumerate(selected_synsets):
                synset_samples = self.synset_groups[synset_id]
                
                # Select samples for this synset
                if len(synset_samples) < samples_per_synset:
                    selected_samples = random.choices(synset_samples, k=samples_per_synset)
                else:
                    selected_samples = random.sample(synset_samples, samples_per_synset)
                
                batch_samples.extend(selected_samples)
                batch_synset_labels.extend([i] * samples_per_synset)
            
            self.batches.append((batch_samples, batch_synset_labels))
    
    def __len__(self):
        return len(self.batches)
    
    def _get_supersense_group(self, supersense: str):
        if supersense.startswith('adj.') or supersense.startswith('adv.'):
            return 1
        elif supersense.startswith('noun.'):
            return 2
        elif supersense.startswith('verb.'):
            return 3
        else:
            return 4
    
    def __getitem__(self, idx):
        samples, synset_labels = self.batches[idx]
        span_indices = [self.span_indices[self.all_samples.index(s)] for s in samples]
        return {
            "samples": samples,
            "synset_labels": synset_labels,
            "span_indices": span_indices
        }

    def custom_collate_fn(self,batch):
        item = batch[0]
        all_samples = item["samples"]
        all_synset_labels = item["synset_labels"]
        all_span_indices = item["span_indices"]

        # Tokenize sentences
        sentences = [s["sentence"] for s in all_samples]
        
        inputs = self.tokenizer(
            sentences, 
            padding=True, 
            truncation=True, 
            max_length=128,
            return_tensors="pt",
            return_attention_mask=True
        )

        
        return {
            "input_ids": inputs["input_ids"],
            "attn_mask": inputs["attention_mask"],
            "span_indices": torch.tensor(all_span_indices),
            "synset_ids": torch.tensor(all_synset_labels)
        }


class CustomSynsetAwareBatchSampler(BatchSampler):
    def __init__(self, dataset, sampler, batch_size, drop_last=False):
        self.dataset = dataset
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

        # Map from synset_id to indices
        self.synset_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.synset_to_indices[sample["synset_id"]].append(idx)

    def __iter__(self):
        sampled_indices = list(self.sampler)
        random.shuffle(sampled_indices)

        i = 0
        while i < len(sampled_indices):
            batch = []
            while len(batch) < self.batch_size:
                if i >= len(sampled_indices):
                    break
                idx = sampled_indices[i]
                i += 1
                synset_id = self.dataset.samples[idx]["synset_id"]

                # Add 50% of the batch from the same synset
                synset_indices = self.synset_to_indices[synset_id]
                same_synset_indices = random.sample(
                    synset_indices,
                    min(len(synset_indices), self.batch_size // 2)
                )

                batch = same_synset_indices.copy()

                # Fill the rest with other random samples
                while len(batch) < self.batch_size and i < len(sampled_indices):
                    rand_idx = sampled_indices[i]
                    if rand_idx not in batch:
                        batch.append(rand_idx)
                    i += 1

                if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
                    yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size