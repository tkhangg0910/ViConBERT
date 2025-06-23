import random
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import PreTrainedTokenizerFast
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize

class PseudoSents_Dataset(Dataset):
    def __init__(self, samples, tokenizer, num_synsets_per_batch=32, samples_per_synset=8, is_training=True,val_mini_batch_size=768):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(self.tokenizer)
        self.num_synsets_per_batch = num_synsets_per_batch
        self.samples_per_synset = samples_per_synset
        self.is_training = is_training  
        self.val_mini_batch_size = val_mini_batch_size
        
        self.synset_groups = defaultdict(list)
        self.synset_to_group = {}
        self.all_samples = []
        self.sample_to_index = {} 

        self.global_synset_to_label = {}
        self.global_label_to_synset = {}

        # Process samples
        all_synsets = set()
        for i, sample in enumerate(tqdm(samples, desc="Processing samples")):
            normalized_sentence = text_normalize(sample["sentence"])
            normalized_target = sample["target_word"]
            group = self._get_supersense_group(sample["supersense"])
            
            new_sample = {
                "sentence": normalized_sentence,
                "target_word": normalized_target,
                "synset_id": sample["synset_id"],
                "group": group
            }
            
            self.synset_groups[sample["synset_id"]].append(new_sample)
            self.synset_to_group[sample["synset_id"]] = group
            self.all_samples.append(new_sample)
            self.sample_to_index[id(new_sample)] = i
            all_synsets.add(sample["synset_id"])

        sorted_synsets = sorted(list(all_synsets))
        for idx, synset_id in enumerate(sorted_synsets):
            self.global_synset_to_label[synset_id] = idx
            self.global_label_to_synset[idx] = synset_id

        # Precompute span indices
        print("Precomputing span indices...")
        self.span_indices = []
        for sample in tqdm(self.all_samples, desc="Computing spans"):
            indices = self.span_extractor.get_span_indices(
                sample["sentence"], 
                sample["target_word"]
            )
            # if not indices 
            if indices:
                pred = self.span_extractor.get_span_text_from_indices(sample["sentence"],indices)
                if sample["target_word"].lower().strip()!= pred.lower().strip():
                    print(f"sentence: {sample['sentence']}")
                    print(f"target: {sample['target_word']}")
                    print(f"pred: {pred}")

            self.span_indices.append(indices if indices else (0, 0))
        
        # Filter synsets with enough samples
        self.valid_synsets = [
            synset_id for synset_id, samples_list in self.synset_groups.items()
            if len(samples_list) >= 1  
        ]
        
        print(f"Total synsets: {len(self.synset_groups)}")
        print(f"Valid synsets: {len(self.valid_synsets)}")
        print(f"Total samples: {len(self.all_samples)}")
        print(f"Global label count: {len(self.global_synset_to_label)}")

        # Generate batches for current epoch
        self._generate_batches()
    
    def _generate_batches(self):
        """Generate batches for one epoch"""
        self.batches = []
        synset_ids = self.valid_synsets.copy()
        
        if self.is_training:
            random.shuffle(synset_ids)
            
            for i in range(0, len(synset_ids), self.num_synsets_per_batch):
                batch_synsets = synset_ids[i:i + self.num_synsets_per_batch]
                
                if len(batch_synsets) < self.num_synsets_per_batch:
                    remaining = self.num_synsets_per_batch - len(batch_synsets)
                    additional = [x for x in self.valid_synsets if x not in batch_synsets]

                    if len(additional) >= remaining:
                        batch_synsets.extend(random.sample(additional, remaining))
                    else:
                        batch_synsets.extend(additional)
                        batch_synsets.extend(random.choices(
                            [x for x in self.valid_synsets if x not in batch_synsets], 
                            k=remaining - len(additional)
                        ))

                batch_samples = []
                batch_synset_labels = []
                
                for label_idx, synset_id in enumerate(batch_synsets):
                    synset_samples = self.synset_groups[synset_id]
                    
                    # Random sampling for training
                    if len(synset_samples) < self.samples_per_synset:
                        selected_samples = random.choices(synset_samples, k=self.samples_per_synset)
                    else:
                        selected_samples = random.sample(synset_samples, self.samples_per_synset)
                    
                    batch_samples.extend(selected_samples)
                    global_label = self.global_synset_to_label[synset_id]
                    batch_synset_labels.extend([global_label] * self.samples_per_synset)
                
                self.batches.append((batch_samples, batch_synset_labels))
        
        else:
            self.batches = []
            batch_size_val = self.val_mini_batch_size

            all_samples = []
            all_synset_labels = []

            for synset_id in synset_ids:
                global_label = self.global_synset_to_label[synset_id]
                for sample in self.synset_groups[synset_id]:
                    all_samples.append(sample)
                    all_synset_labels.append(global_label)
            
            num_samples = len(all_samples)
            for start_idx in range(0, num_samples, self.val_mini_batch_size):
                end_idx = min(start_idx + self.val_mini_batch_size, num_samples)
                batch_samples = all_samples[start_idx:end_idx]
                batch_synset_labels = all_synset_labels[start_idx:end_idx]
                self.batches.append((batch_samples, batch_synset_labels))
            
            print(f"Validation batches: {len(self.batches)}")
            print(f"Total validation samples: {num_samples}")


    
    def __len__(self):
        return len(self.batches)
    
    def _get_supersense_group(self, supersense: str):
        """Map supersense to group"""
        if supersense.startswith(('adj.', 'adv.')):
            return 1
        elif supersense.startswith('noun.'):
            return 2
        elif supersense.startswith('verb.'):
            return 3
        else:
            return 4
    
    def __getitem__(self, idx):
        """Return a single batch"""
        samples, synset_labels = self.batches[idx]
        
        span_indices = []
        for sample in samples:
            sample_idx = self.sample_to_index[id(sample)]
            span_indices.append(self.span_indices[sample_idx])
        
        return {
            "samples": samples,
            "synset_labels": synset_labels,
            "span_indices": span_indices
        }
    
    def on_epoch_end(self):
        """Call this at the end of each epoch to regenerate batches (only for training)"""
        if self.is_training:
            self._generate_batches()

    def custom_collate_fn(self, batch):
        """Collate function for DataLoader"""
        if len(batch) != 1:
            raise ValueError("Batch size should be 1 when using custom_collate_fn")
            
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
            "span_indices": torch.tensor(all_span_indices, dtype=torch.long),
            "synset_ids": torch.tensor(all_synset_labels, dtype=torch.long)
        }



# class CustomSynsetAwareBatchSampler(BatchSampler):
#     def __init__(self, dataset, sampler, batch_size, drop_last=False):
#         self.dataset = dataset
#         self.sampler = sampler
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#         # Map from synset_id to indices
#         self.synset_to_indices = defaultdict(list)
#         for idx, sample in enumerate(dataset.samples):
#             self.synset_to_indices[sample["synset_id"]].append(idx)

#     def __iter__(self):
#         sampled_indices = list(self.sampler)
#         random.shuffle(sampled_indices)

#         i = 0
#         while i < len(sampled_indices):
#             batch = []
#             while len(batch) < self.batch_size:
#                 if i >= len(sampled_indices):
#                     break
#                 idx = sampled_indices[i]
#                 i += 1
#                 synset_id = self.dataset.samples[idx]["synset_id"]

#                 # Add 50% of the batch from the same synset
#                 synset_indices = self.synset_to_indices[synset_id]
#                 same_synset_indices = random.sample(
#                     synset_indices,
#                     min(len(synset_indices), self.batch_size // 2)
#                 )

#                 batch = same_synset_indices.copy()

#                 # Fill the rest with other random samples
#                 while len(batch) < self.batch_size and i < len(sampled_indices):
#                     rand_idx = sampled_indices[i]
#                     if rand_idx not in batch:
#                         batch.append(rand_idx)
#                     i += 1

#                 if len(batch) == self.batch_size or (not self.drop_last and len(batch) > 0):
#                     yield batch

#     def __len__(self):
#         if self.drop_last:
#             return len(self.sampler) // self.batch_size
#         else:
#             return (len(self.sampler) + self.batch_size - 1) // self.batch_size