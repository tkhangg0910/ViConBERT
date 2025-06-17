from torch.utils.data import Dataset, WeightedRandomSampler
from collections import defaultdict
from models.base_model import SpanExtractor
from transformers import AutoTokenizer
import torch

class PseudoSents_Dataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.word_to_synsets = defaultdict(list)
        self.synset_groups = defaultdict(list)
        self.word_group_pairs = defaultdict(list)
        self.synset_to_group = {}
        self.polysemous_words = []

        # Build dictionaries and identify polysemous words
        for sample in samples:
            synset_id = sample["synset_id"]
            word = sample["target_word"]
            supersense = sample["supersense"]
            group = self._get_supersense_group(supersense)

            self.synset_groups[synset_id].append(sample)
            self.word_to_synsets[word].append((synset_id, group))
            self.word_group_pairs[(word, group)].append(sample)
            self.synset_to_group[synset_id] = group

        # Identify polysemous words (words with >1 unique supersense group)
        for word, synsets in self.word_to_synsets.items():
            unique_groups = {group for _, group in synsets}
            if len(unique_groups) > 1:
                self.polysemous_words.append(word)

        # Compute sample weights
        self.sample_weights = []
        max_group_size = max(len(g) for g in self.synset_groups.values()) if self.synset_groups else 1
        
        for sample in self.samples:
            synset_id = sample["synset_id"]
            word = sample["target_word"]
            group_size = len(self.synset_groups[synset_id])
            base_weight = max_group_size / group_size  
            weight = base_weight * 2 if word in self.polysemous_words else base_weight
            self.sample_weights.append(weight)

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
            "synset_id": sample["synset_id"]
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
    
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    inputs = tokenizer(
        sentences, 
        padding=True, 
        truncation=True, 
        return_tensors="pt",
        max_length=512
    )
    span_extractor = SpanExtractor(tokenizer)
    span_indices = []
    for text, target in zip(sentences, target_words):
        indices = span_extractor.get_span_indices(text, target)
        span_indices.append(indices if indices else (0, 0))
    
    return {
        "input_ids": inputs["input_ids"],
        "attn_mask":inputs["attention_mask"],
        "span_indices": torch.tensor(span_indices),
        "synset_ids": torch.tensor(synset_ids)
    }
     
