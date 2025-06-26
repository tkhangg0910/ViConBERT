import re
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from collections import defaultdict
from utils.process_data import text_normalize
from utils.span_extractor import SpanExtractor,SentenceMasking

class SuperSenseDataset(Dataset):
    def __init__(self, samples, tokenizer,
                 use_sent_masking=False,
                 training=True):
        self.tokenizer = tokenizer
        self.use_sent_masking= use_sent_masking
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if use_sent_masking:
            self.sent_masking = SentenceMasking(self.tokenizer)
        else:
            self.span_extractor = SpanExtractor(self.tokenizer)
        self.training = training
        self.all_samples = []
        self.sample_to_index = {} 
        self.supersense_groups = defaultdict(list)
        self.global_supersense_to_label = {}
        self.global_label_to_supersense = {}
        if not training:
            self.global_synset_to_label = {}
            self.global_label_to_synset = {}
            all_synsets = set()
            
        # Process samples
        all_supersense = set()
        for i, sample in enumerate(tqdm(samples, desc="Processing samples", ascii=True)):
            normalized_sentence = text_normalize(sample["sentence"])
            normalized_target = sample["target_word"]
            
            new_sample = {
                "sentence": normalized_sentence,
                "target_word": normalized_target,
                "synset_id": sample["synset_id"],
                "supersense":sample["supersense"]
            }
            self.supersense_groups[sample["supersense"]].append(new_sample)
            self.all_samples.append(new_sample)
            self.sample_to_index[id(new_sample)] = i
            all_supersense.add(sample["supersense"])
            if not training:
                all_synsets.add(sample["synset_id"])
                
        if not training:   
            sorted_synsets = sorted(list(all_synsets))
            for idx, synset_id in enumerate(sorted_synsets):
                self.global_synset_to_label[synset_id] = idx
                self.global_label_to_synset[idx] = synset_id

        sorted_supersense = sorted(list(all_supersense))
        for idx, supersense in enumerate(sorted_supersense):
            self.global_supersense_to_label[supersense] = idx
            self.global_label_to_supersense[idx] = supersense

        # Precompute span indices
        if use_sent_masking:
            print("Precomputing masking sentence...")
            self.masked_sents = []
            for sample in tqdm(self.all_samples, desc="Computing spans",ascii=True):
                masked_sent, span_idx = self.sent_masking.create_masked_version(
                    sample["sentence"], 
                    sample["target_word"]
                )
                if "<mask>" not in masked_sent.split() and not re.search(r"<mask>", masked_sent):
                    print(masked_sent)
                self.masked_sents.append(masked_sent)
        else:
            print("Precomputing span indices...")
            self.span_indices = []
            for sample in tqdm(self.all_samples, desc="Computing spans",ascii=True):
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
        
        
        print(f"Total supersense: {len(self.supersense_groups)}")
        print(f"Total samples: {len(self.all_samples)}")
        print(f"Global label count: {len(self.global_supersense_to_label)}")


    
    def __len__(self):
        """Returns total number of samples in dataset"""
        return len(self.all_samples)
        
    def __getitem__(self, idx):
        """Return a single sample with all necessary components"""
        sample = self.all_samples[idx]
        
        item = {
            "sample": sample,
            "supersense_label": self.global_supersense_to_label[sample["supersense"]],
            "synset_id": sample["synset_id"] if self.training else self.global_synset_to_label[sample["synset_id"]]
        }
        
        if self.use_sent_masking:
            item["masked_sent"] = self.masked_sents[idx]  
        else:
            item["span_indices"] = self.span_indices[idx]
        
        return item

    def collate_fn(self, batch):
        """Custom collate function to process a batch of samples"""
        target_word = [item["sample"]['target_word'] for item in batch]
        target_spans = [item['span_indices'] for item in batch ]
        synset_ids = [item["synset_id"] for item in batch]
        supersense_labels = [item["supersense_label"] for item in batch]
        
        
        
        word_inputs = self.tokenizer(
            target_word,
            padding=True, 
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
            
        if self.use_sent_masking:
            masked_sent = [item['masked_sent'] for item in batch]
            context_inputs = self.tokenizer(
                masked_sent,
                padding=True ,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            target_spans = None
        else:
            sentences = [item["sample"]["sentence"] for item in batch]
            context_inputs = self.tokenizer(
                sentences,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            target_spans = [item['span_indices'] for item in batch ]
        
        
            

        
        padded_batch = {
            "target_spans":target_spans,
            "context_input_ids": context_inputs["input_ids"],
            "context_attn_mask": context_inputs["attention_mask"],
            "word_input_ids": word_inputs["input_ids"],
            "word_attn_mask": word_inputs["attention_mask"],
            "synset_ids": torch.tensor(synset_ids, dtype=torch.long),
            "supersense_labels": torch.tensor(supersense_labels, dtype=torch.long)
        }
        
        return padded_batch