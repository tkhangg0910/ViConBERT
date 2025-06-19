import random
from torch.utils.data import Dataset, WeightedRandomSampler
from collections import defaultdict
from transformers import PreTrainedTokenizerFast
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize

class PseudoSents_Dataset(Dataset):
    def __init__(self, samples, tokenizer):
        self.samples = samples
        self.word_to_synsets = defaultdict(list)
        self.synset_groups = defaultdict(list)
        self.word_group_pairs = defaultdict(list)
        self.synset_to_group = {}
        self.polysemous_words = []

        print("Normalizing text...")
        sentences = [s["sentence"] for s in samples]
        target_words = [s["target_word"] for s in samples]

        normalized_sentences = [text_normalize(s) for s in tqdm(sentences)]
        normalized_targets = [tw for tw in tqdm(target_words)]

        for i, sample in enumerate(samples):
            sample["sentence"] = normalized_sentences[i]
            sample["target_word"] = normalized_targets[i]
        
        # Build dictionaries and identify polysemous words
        print("Building dictionaries...")
        for sample in tqdm(samples):
            synset_id = sample["synset_id"]
            word = sample["target_word"]
            supersense = sample["supersense"]
            group = self._get_supersense_group(supersense)

            self.synset_groups[synset_id].append(sample)
            self.word_to_synsets[word].append((synset_id, group))
            self.word_group_pairs[(word, group)].append(sample)
            self.synset_to_group[synset_id] = group

        # Identify polysemous words (words with >1 unique supersense group)
        print("Identifying polysemous words...")
        for word, synsets in tqdm(self.word_to_synsets.items()):
            unique_groups = {group for _, group in synsets}
            if len(unique_groups) > 1:
                self.polysemous_words.append(word)

        # Compute sample weights
        print("Computing sample weights...")
        self.sample_weights = []
        max_group_size = max(len(g) for g in self.synset_groups.values()) if self.synset_groups else 1
        
        for sample in tqdm(samples):
            synset_id = sample["synset_id"]
            word = sample["target_word"]
            group_size = len(self.synset_groups[synset_id])
            base_weight = max_group_size / group_size  
            weight = base_weight * 2 if word in self.polysemous_words else base_weight
            self.sample_weights.append(weight)
        print("Precomputing span indices...")
        
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(self.tokenizer)

        self.span_indices = []
        for sample in tqdm(samples):
            text = sample["sentence"]
            target = sample["target_word"]
            indices = self.span_extractor.get_span_indices(text, target)
            if indices:
                pred = self.span_extractor.get_span_text_from_indices(text,indices)
                if target.lower().strip()!= pred.lower().strip():
                    print(f"sentence: {text}")
                    print(f"target: {target}")
                    print(f"pred: {pred}")
            self.span_indices.append(indices if indices else (0, 0))

    def __len__(self):
        return len(self.samples)
    
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
        sample = self.samples[idx]
        return {
            "sentence": sample["sentence"],
            "target_word": sample["target_word"],
            "synset_id": sample["synset_id"],
            "span_indices": self.span_indices[idx],
            "tokenizer": self.tokenizer
        }

    def get_weighted_sampler(self):
        weights = torch.tensor(self.sample_weights, dtype=torch.float)
        return WeightedRandomSampler(
            weights, 
            num_samples=len(self), 
            replacement=True
        )

def custom_collate_fn(batch):
    sentences = [item["sentence"] for item in batch]
    target_words = [item["target_word"] for item in batch]
    synset_ids = [item["synset_id"] for item in batch]
    span_indices = [item["span_indices"] for item in batch]

    
    tokenizer = batch[0]["tokenizer"] if "tokenizer" in batch[0] else \
        PreTrainedTokenizerFast.from_pretrained("vinai/phobert-base")

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    inputs = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=True,
        max_length=512
    )
    

    return {
        "input_ids": inputs["input_ids"],
        "attn_mask": inputs["attention_mask"],
        "span_indices": torch.tensor(span_indices),
        "synset_ids": torch.tensor(synset_ids)
    }

     
class ProportionalBatchSampler:
    def __init__(self,dataset, batch_size, positive_ratio=0.3, min_positive_samples=2):
        """
        Args:
            positive_ratio: proportion of positive samples in batch (0.0 - 1.0)
            min_positive_samples: min number of positive samples in a batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.positive_ratio = positive_ratio
        self.min_positive_samples = min_positive_samples
        
        self.positive_samples_per_batch = max(
            min_positive_samples, 
            int(batch_size * positive_ratio)
        )

        self.synset_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            synset_id = sample["synset_id"]
            self.synset_to_indices[synset_id].append(idx)

        self.valid_synsets = [
            synset for synset, indices in self.synset_to_indices.items()
            if len(indices) >= min_positive_samples
        ]

        self.positive_groups = []

        for synset in self.valid_synsets:
            indices = self.synset_to_indices[synset]
            random.shuffle(indices)
            
            for i in range(0, len(indices), min_positive_samples):
                group = indices[i:i+min_positive_samples]
                if len(group) == min_positive_samples:
                    self.positive_groups.append(group)
                    
        self.all_indices = list(range(len(dataset)))
        random.shuffle(self.all_indices)
        
        self.num_batches = len(self.all_indices) // (
            batch_size - self.positive_samples_per_batch
        )

    def __iter__(self):
        random.shuffle(self.positive_groups)
        positive_group_iter = iter(self.positive_groups)
        
        random.shuffle(self.all_indices)
        all_indices_iter = iter(self.all_indices)
        
        for _ in range(self.num_batches):
            batch_indices = []
            
            try:
                for _ in range(self.positive_samples_per_batch // self.min_positive_samples):
                    batch_indices.extend(next(positive_group_iter))
            except StopIteration:
                pass
            
            remaining = self.batch_size - len(batch_indices)
            for _ in range(remaining):
                try:
                    batch_indices.append(next(all_indices_iter))
                except StopIteration:
                    break
            
            yield batch_indices

    def __len__(self):
        return self.num_batches
