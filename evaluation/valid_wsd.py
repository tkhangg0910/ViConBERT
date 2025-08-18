import argparse
import json
import os
import torch
from transformers import PhobertTokenizerFast, XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
from data.processed.polybert_exp.dataset import PolyBERTtDatasetV3
from models.base_model import ViSynoSenseEmbedding
from utils.load_config import load_config
import torch
from tqdm import tqdm
from transformers.utils import is_torch_available
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
import numpy as np
from utils.span_extractor import SpanExtractor
from utils.process_data import text_normalize
from collections import defaultdict

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--batch_size", type=int,default=768, help="Batch size")
    args = parser.parse_args()
    return args 
import torch.nn.functional as F
from pyvi.ViTokenizer import tokenize
        
class WSD_ViConDataset(Dataset):
    def __init__(self, samples, tokenizer, gloss_emd, val_mode=False):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)
        self.gloss_emd = gloss_emd
        self.all_samples = []
        self.word2gloss = defaultdict(list)
        self.val_mode = val_mode

        for sample in samples:
            sent = text_normalize(sample["sentence"])
            item = {
                "sentence": sent,
                "target_word": sample["target_word"],
                "synset_id": sample["synset_id"],
                "gloss": sample["gloss"],
                "gloss_id": sample["gloss_id"],
                "word_id": int(sample["word_id"])
            }
            self.all_samples.append(item)
            self.word2gloss[item["target_word"]].append(item)

        # span indices
        self.span_indices = []
        for s in tqdm(self.all_samples, desc="Computing spans", ascii=True):
            idxs = self.span_extractor.get_span_indices(
                s["sentence"], s["target_word"]
            )
            self.span_indices.append(idxs or (0,0))

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        s = self.all_samples[idx]
        item = {
            "sentence": s["sentence"],
            "target_word": s["target_word"],
            "target_spans": self.span_indices[idx],
            "gloss": s["gloss"],
            "gloss_id": s["gloss_id"],
            "word_id": s["word_id"],
            "synset_id": s["synset_id"]
        }

        # candidate gloss IDs và vector embedding
        if self.val_mode:
            candidate_samples = self.word2gloss[s["target_word"]]
            # Lấy unique gloss_id
            seen = set()
            unique_candidates = []
            for c in candidate_samples:
                if c["gloss_id"] not in seen:
                    seen.add(c["gloss_id"])
                    unique_candidates.append(c)

            candidate_ids = [c["gloss_id"] for c in unique_candidates]
            candidate_vectors = [self.gloss_emd[cid] for cid in candidate_ids]

            item["candidate_gloss_ids"] = candidate_ids
            item["candidate_gloss_vectors"] = candidate_vectors


        return item

    def collate_fn(self, batch):
        sentences = [b["sentence"] for b in batch]  
        spans = [b["target_spans"] for b in batch]  
        synset_ids = torch.tensor([b["synset_id"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
        target_words = [b["target_word"] for b in batch]
        gold_glosses = [b["gloss"] for b in batch]
        gloss_id = torch.tensor([b["gloss_id"] for b in batch], dtype=torch.long)

        # Candidate gloss vectors
        candidate_vectors_batch = []
        candidate_ids_batch = []
        for b in batch:
            if self.val_mode:
                candidate_vectors_batch.append(torch.stack(b["candidate_gloss_vectors"]))
                candidate_ids_batch.append(torch.tensor(b["candidate_gloss_ids"], dtype=torch.long))
            else:
                candidate_vectors_batch.append(torch.tensor([]))
                candidate_ids_batch.append(torch.tensor([]))

        return {
            "sentence": sentences,
            "target_spans": torch.tensor(spans, dtype=torch.long),
            "synset_ids": synset_ids,
            "word_id": word_id,
            "target_words": target_words,
            "gloss": gold_glosses,
            "gloss_id": gloss_id,
            "candidate_gloss_vectors": candidate_vectors_batch,
            "candidate_gloss_ids": candidate_ids_batch
        }

def evaluate_model(context_model, data_loader, device):
    valid_loss = 0.0
    TP, FP, FN = 0, 0, 0
    valid_steps = 0

    context_model.eval()
    with torch.no_grad():
        val_pbar = tqdm(data_loader, desc=f"Validating", position=1, leave=True, ascii=True)
        for batch in val_pbar:
            c_toks = context_model.tokenizer(
                batch["sentence"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            ).to(device)

            target_spans = batch["target_spans"].to(device)
            rF_wt = context_model(
                {
                "input_ids":c_toks["input_ids"],
                "attention_mask":c_toks["attention_mask"],
                    },
                target_spans
            )  # [B, polym, H]

            for i in range(len(batch)):
                gold_gloss_id = batch["gloss_id"][i].item()
                rF_g = batch["candidate_gloss_vectors"][i].to(device)  # [N, H]

                # similarity context_i vs N candidate glosses
                sim = torch.matmul(
                    rF_wt[i].flatten().unsqueeze(0),  # [1, H]
                    rF_g.T  # [H, N]
                )  # [1, N]

                P = F.softmax(sim, dim=1)  # [1, N]

                candidate_ids = batch["candidate_gloss_ids"][i].to(device)  # [N]
                gold_idx = (candidate_ids == gold_gloss_id).nonzero(as_tuple=True)[0].item()

                loss_i = -torch.log(P[0, gold_idx] + 1e-8)
                valid_loss += loss_i.item()
                valid_steps += 1

                # prediction
                pred_idx = P.argmax(dim=1).item()
                if pred_idx == gold_idx:
                    TP += 1
                else:
                    FP += 1
                    FN += 1

    valid_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    valid_recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    valid_f1        = 2 * valid_precision * valid_recall / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0.0
    avg_valid_loss  = valid_loss / max(1, valid_steps)

    return {
        "loss": avg_valid_loss,
        "valid_f1": valid_f1,
        "valid_recall": valid_recall,
        "valid_precision": valid_precision
    }

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(42) 
    args = setup_args()
    
    config = load_config("configs/poly.yml")
    
    with open(config["data"]["valid_path"], "r", encoding="utf-8") as f:
        valid_sample = json.load(f)
        
    tokenizer = PhobertTokenizerFast.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load gloss embeddings
    gloss_emd = torch.load("gloss_embeddings.pt")  # {gloss_id: tensor}
    
    # Dùng dataset đã chỉnh sửa
    valid_set = WSD_ViConDataset(valid_sample, tokenizer, gloss_emd, val_mode=True)
    
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_set.collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    context_model = ViSynoSenseEmbedding.from_pretrained(args.model_path).to(device)
    
    print("\nValidating epoch...")
    valid_metrics = evaluate_model(context_model, valid_dataloader, device)
    
    print("\n  VALIDATION METRICS:")
    print(f"    Loss: {valid_metrics['loss']:.4f}")
    print(f"    F1: {valid_metrics['valid_f1']:.4f}")
    print(f"    Recall: {valid_metrics['valid_recall']:.4f}")
    print(f"    Precision: {valid_metrics['valid_precision']:.4f}")
