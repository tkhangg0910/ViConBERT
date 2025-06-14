from torch.utils.data import Dataset
from collections import defaultdict
from models.base_model import SpanExtractor
from transformers import AutoTokenizer
import torch

class PseudoSents_Dataset(Dataset):
    def __init___(self, samples):
        self.samples = samples

        self.synset_groups = defaultdict(list)

        for sample in samples:
            self.synset_groups[sample["synset_id"]].append(sample)
        
        self.balanced_samples = []

        max_samples = max(len(g) for g in self.synset_groups.values())

        for synset_id, group in self.synset_groups.items():
            repeated_group = group * (max_samples // len(group)) + group[:max_samples % len(group)]
            self.balanced_samples.extend(repeated_group)
            
    def __len__(self):
        return len(self.balanced_samples)
    
    def __getitem__(self, idx):
        sample = self.balanced_samples[idx]
        return {
            "sentence": sample["sentence"],
            "target_word": sample["target_word"],
            "synset_id": sample["synset_id"]
        }

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
     
