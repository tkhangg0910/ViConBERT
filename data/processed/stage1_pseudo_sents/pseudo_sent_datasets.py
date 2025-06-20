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
                # pred = self.span_extractor.get_span_text_from_indices(text,indices)
                # if target.lower().strip()!= pred.lower().strip():
                #     print(f"sentence: {text}")
                #     print(f"target: {target}")
                #     print(f"pred: {pred}")
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


    inputs = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        add_special_tokens=True,
        max_length=258 ,
        return_token_type_ids=False,  
        return_attention_mask=True,
        return_offsets_mapping=False   
    )
    if "token_type_ids" in inputs:
        del inputs["token_type_ids"]


    return {
        "input_ids": inputs["input_ids"],
        "attn_mask": inputs["attention_mask"],
        "span_indices": torch.tensor(span_indices),
        "synset_ids": torch.tensor(synset_ids)
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
