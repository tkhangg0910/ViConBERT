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


        for sample in samples:
            synset_id = sample["synset_id"]
            word = sample["target_word"]
            supersense = sample["supersense"]
            group = self._get_supersense_group(supersense)

            self.synset_groups[synset_id].append(sample)
            self.word_to_synsets[word].append((synset_id, group))
            self.word_group_pairs[(word, group)].append(sample)
            self.synset_to_group[synset_id] = group

        self.polysemous_words = []
        self.synset_polysemy_weights = defaultdict(int)  
        
        for word, synsets in self.word_to_synsets.items():
            unique_groups = {group for _, group in synsets}
            
            if len(unique_groups) > 1:
                self.polysemous_words.append(word)
                
            for synset_id, _ in synsets:
                    self.synset_polysemy_weights[synset_id] += 1

        self.balanced_samples = []
        self.sample_weights = []
        
        max_samples = max(len(g) for g in self.synset_groups.values()) if self.synset_groups else 0

        for synset_id, group_samples in self.synset_groups.items():
            polysemy_weight = max(1, self.synset_polysemy_weights[synset_id])  
            
            repeat_count = max_samples // len(group_samples) * polysemy_weight
            remainder = max_samples % len(group_samples)

            repeated_samples = group_samples * repeat_count + group_samples[:remainder * polysemy_weight]
            self.balanced_samples.extend(repeated_samples)

            weight_val = 2.0 if polysemy_weight > 1 else 1.0
            self.sample_weights.extend([weight_val] * len(repeated_samples))

            
    def __len__(self):
        return len(self.balanced_samples)
    
        
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
        sample = self.balanced_samples[idx]
        return {
            "sentence": sample["sentence"],
            "target_word": sample["target_word"],
            "synset_id": sample["synset_id"]
        }

    def get_weighted_sampler(self):
        """Tạo WeightedSampler cho việc lấy mẫu ưu tiên"""
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
     
