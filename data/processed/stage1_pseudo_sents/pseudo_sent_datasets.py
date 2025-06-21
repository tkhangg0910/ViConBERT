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
    def __init__(self, samples, tokenizer, num_synsets_per_batch=32, samples_per_synset=8, is_training=True):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(self.tokenizer)
        self.num_synsets_per_batch = num_synsets_per_batch
        self.samples_per_synset = samples_per_synset
        self.is_training = is_training  # Phân biệt training vs validation

        self.synset_groups = defaultdict(list)
        self.synset_to_group = {}
        self.all_samples = []
        self.sample_to_index = {}  # Mapping để tối ưu hiệu suất

        # Process samples
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
            
        # Precompute span indices
        print("Precomputing span indices...")
        self.span_indices = []
        for sample in tqdm(self.all_samples, desc="Computing spans"):
            indices = self.span_extractor.get_span_indices(
                sample["sentence"], 
                sample["target_word"]
            )
            self.span_indices.append(indices if indices else (0, 0))
        
        # Filter synsets with enough samples
        self.valid_synsets = [
            synset_id for synset_id, samples_list in self.synset_groups.items()
            if len(samples_list) >= 1  # Có thể adjust threshold này
        ]
        
        print(f"Total synsets: {len(self.synset_groups)}")
        print(f"Valid synsets: {len(self.valid_synsets)}")
        print(f"Total samples: {len(self.all_samples)}")
        
        # Generate batches for current epoch
        self._generate_batches()
    
    def _generate_batches(self):
        """Generate batches for one epoch"""
        self.batches = []
        synset_ids = self.valid_synsets.copy()
        
        if self.is_training:
            # Training: shuffle và random sampling
            random.shuffle(synset_ids)
            
            # Tạo batches, đảm bảo sử dụng tất cả synsets
            for i in range(0, len(synset_ids), self.num_synsets_per_batch):
                batch_synsets = synset_ids[i:i + self.num_synsets_per_batch]
                
                # Nếu batch cuối không đủ synsets, padding với random synsets
                if len(batch_synsets) < self.num_synsets_per_batch:
                    remaining = self.num_synsets_per_batch - len(batch_synsets)
                    batch_synsets.extend(random.choices(self.valid_synsets, k=remaining))
                
                batch_samples = []
                batch_synset_labels = []
                
                for label_idx, synset_id in enumerate(batch_synsets):
                    synset_samples = self.synset_groups[synset_id]
                    
                    # Random sampling cho training
                    if len(synset_samples) < self.samples_per_synset:
                        selected_samples = random.choices(synset_samples, k=self.samples_per_synset)
                    else:
                        selected_samples = random.sample(synset_samples, self.samples_per_synset)
                    
                    batch_samples.extend(selected_samples)
                    batch_synset_labels.extend([label_idx] * self.samples_per_synset)
                
                self.batches.append((batch_samples, batch_synset_labels))
        
        else:
            # Validation: deterministic, evaluate toàn bộ tập
            # Không shuffle, giữ thứ tự cố định
            synset_ids.sort()  # Đảm bảo thứ tự cố định
            
            for i in range(0, len(synset_ids), self.num_synsets_per_batch):
                batch_synsets = synset_ids[i:i + self.num_synsets_per_batch]
                
                # Với validation, nếu batch cuối không đủ synsets thì cứ để vậy
                # Không cần padding để tránh evaluate trùng lặp
                batch_samples = []
                batch_synset_labels = []
                
                for label_idx, synset_id in enumerate(batch_synsets):
                    synset_samples = self.synset_groups[synset_id]
                    
                    # Lấy tất cả samples hoặc cố định số lượng
                    if self.samples_per_synset == -1:
                        # Lấy tất cả samples
                        selected_samples = synset_samples
                        samples_count = len(selected_samples)
                    else:
                        # Lấy số lượng cố định, nhưng deterministic
                        if len(synset_samples) <= self.samples_per_synset:
                            selected_samples = synset_samples
                        else:
                            # Lấy samples đầu tiên (deterministic)
                            selected_samples = synset_samples[:self.samples_per_synset]
                        samples_count = len(selected_samples)
                    
                    batch_samples.extend(selected_samples)
                    batch_synset_labels.extend([label_idx] * samples_count)
                
                if batch_samples:  # Chỉ add batch nếu có samples
                    self.batches.append((batch_samples, batch_synset_labels))
    
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
        
        # Tối ưu hóa việc lấy span indices
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
        # Validation dataset không cần regenerate vì cần kết quả consistent

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