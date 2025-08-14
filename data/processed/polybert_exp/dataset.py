import random
import re
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize
from torch.utils.data import Sampler
from collections import defaultdict
import math

class PolyBERTtDataset(Dataset):
    def __init__(self, samples, tokenizer, val_mode = False):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)
        self.val_mode =val_mode
        if self.val_mode:
            self.targetword2glosses = defaultdict(list)
        word_set = set()
        self.all_samples = []
        for sample in samples:
            sent = text_normalize(sample["sentence"])
            word_id = sample["word_id"]
            target = sample["target_word"]
            sid = sample["synset_id"]
            gloss=sample["gloss"]
            self.all_samples.append({
                "sentence": sent,
                "target_word": target,
                "synset_id": sid,
                "gloss":gloss,
                "word_id":int(word_id)
            })
            
            if self.val_mode:
                if gloss not in self.targetword2glosses[target]:
                    self.targetword2glosses[target].append(gloss)

            word_set.add(int(word_id))
            
        sorted_wids = sorted(word_set)
        self.global_word_to_label = {wid: i for i, wid in enumerate(sorted_wids)}

        self.span_indices = []
        for s in tqdm(self.all_samples, desc="Computing spans",ascii=True):
            idxs = self.span_extractor.get_span_indices(
                s["sentence"], s["target_word"]
            )
            # if idxs:
            #     pred = self.span_extractor.get_span_text_from_indices(s["sentence"],idxs)
            #     if s["target_word"].lower().strip()!= pred.lower().strip():
            #         print(f"sentence: {s['sentence']}")
            #         print(f"target: {s['target_word']}")
            #         print(f"pred: {pred}")
            self.span_indices.append(idxs or (0,0))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s = self.all_samples[idx]
        sid = s["synset_id"]
        wid = s["word_id"]
        label = self.global_word_to_label[int(wid)]
        item = {
            "sentence": s["sentence"],
            "target_span": self.span_indices[idx],
            "word_id": label,
            "target_word": s["target_word"],
            "gloss" : s["gloss"],
            "synset_ids":sid
        }
        if self.val_mode:
            # candidate glosses = all glosses observed for this target_word
            item["candidate_glosses"] = list(self.targetword2glosses[s["target_word"]])
        return item

    def collate_fn(self, batch):
        if self.val_mode:
            # batch: list of items (each item has candidate_glosses list)
            sentences = [b["sentence"] for b in batch]          # B
            spans     = [b["target_span"] for b in batch]       # B x (2)
            synset_ids = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
            word_id = torch.tensor([(b["word_id"]) for b in batch], dtype=torch.long)
            target_words = [b["target_word"] for b in batch]
            gold_glosses = [b["gloss"] for b in batch]

            # tokenise contexts (B)
            c_toks = self.tokenizer(
                sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            )

            candidate_glosses = [b["candidate_glosses"] for b in batch]
        
            return {
                "context_input_ids": c_toks["input_ids"],           # [B, Lc]
                "context_attn_mask": c_toks["attention_mask"],      # [B, Lc]
                "target_spans": torch.tensor(spans, dtype=torch.long),  # [B,2]
                "synset_ids": synset_ids,
                "word_id": word_id,
                "target_words": target_words,
                "gold_glosses": gold_glosses,                
                "candidate_glosses": candidate_glosses, 
            }

        sentences = [b["sentence"] for b in batch]
        target_words = [b["target_word"] for b in batch]
        spans     = [b["target_span"] for b in batch]
        labels    = torch.tensor([b["synset_ids"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
        glosses = [b["gloss"] for b in batch]

        c_toks = self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        g_tokes = self.tokenizer(
            glosses,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
            return_attention_mask=True,
            return_offsets_mapping=True
        )
        return {
            "context_input_ids": c_toks["input_ids"],
            "context_attn_mask": c_toks["attention_mask"],
            "target_spans": torch.tensor(spans, dtype=torch.long) if spans else None,
            "synset_ids": labels,
            "word_id":word_id,
            "target_words":target_words,
            "gloss_input_ids": g_tokes["input_ids"],
            "gloss_attn_mask": g_tokes["attention_mask"],
        }


class ContrastiveBatchSampler(Sampler):
    def __init__(self, dataset, min_senses=2, samples_per_sense=2, batch_size=64, drop_last=False):
        """
        Args:
            dataset (PolyBERTtDataset): Dataset đã được khởi tạo
            min_senses (int): Số nghĩa tối thiểu mỗi từ phải có
            samples_per_sense (int): Số mẫu lấy cho mỗi nghĩa
            batch_size (int): Kích thước batch mong muốn
            drop_last (bool): Có bỏ batch cuối nếu không đủ size
        """
        self.dataset = dataset
        self.min_senses = min_senses
        self.samples_per_sense = samples_per_sense
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Validate batch size
        self.words_per_batch = batch_size // (min_senses * samples_per_sense)
        assert self.words_per_batch > 0, "batch_size quá nhỏ so với min_senses * samples_per_sense"
        
        # Tạo cấu trúc dữ liệu phân cấp
        self.word_to_senses = defaultdict(lambda: defaultdict(list))
        for idx, sample in enumerate(dataset.all_samples):
            word = sample["target_word"]
            sense_id = sample["word_id"]
            self.word_to_senses[word][sense_id].append(idx)
        
        # Lọc từ có đủ nghĩa
        self.eligible_words = [
            word for word, senses in self.word_to_senses.items()
            if len(senses) >= min_senses
        ]
        
        # Tính số batch
        self.num_batches = len(self.eligible_words) // self.words_per_batch
        if not drop_last and len(self.eligible_words) % self.words_per_batch != 0:
            self.num_batches += 1

    def __iter__(self):
        # Xáo trộn từ đủ điều kiện
        random.shuffle(self.eligible_words)
        
        # Tạo batches
        for batch_idx in range(self.num_batches):
            start_idx = batch_idx * self.words_per_batch
            end_idx = start_idx + self.words_per_batch
            
            # Xử lý batch cuối
            if batch_idx == self.num_batches - 1 and end_idx > len(self.eligible_words):
                if self.drop_last:
                    continue
                end_idx = len(self.eligible_words)
            
            batch_words = self.eligible_words[start_idx:end_idx]
            batch_indices = []
            
            for word in batch_words:
                # Lấy tất cả nghĩa của từ
                senses = list(self.word_to_senses[word].keys())
                
                # Ưu tiên chọn các nghĩa khác nhau
                selected_senses = random.sample(
                    senses, 
                    k=min(self.min_senses, len(senses))
                )
                for sense_id in selected_senses:
                    # Lấy mẫu cho nghĩa này
                    sense_samples = self.word_to_senses[word][sense_id]
                    
                    # Xử lý khi không đủ mẫu
                    if len(sense_samples) < self.samples_per_sense:
                        selected = random.choices(sense_samples, k=self.samples_per_sense)
                    else:
                        selected = random.sample(sense_samples, k=self.samples_per_sense)
                    
                    batch_indices.extend(selected)
            
            yield batch_indices

    def __len__(self):
        return self.num_batches 