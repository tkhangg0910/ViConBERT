import argparse
import json
import os
import torch
from transformers import PhobertTokenizerFast, XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
from data.processed.polybert_exp.dataset import PolyBERTtDatasetV3
from models.base_model import ViSynoSenseEmbedding
from models.polybert import PolyBERT
from models.bem import BiEncoderModel
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
import torch.nn.functional as F
from pyvi.ViTokenizer import tokenize

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
def setup_args():
    parser = argparse.ArgumentParser(description="Evaluate WSD model")
    parser.add_argument("--model_path", type=str, required=True, help="Context model path")
    parser.add_argument("--model_type", type=str, required=True, help="Context model path")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    
    # Two modes
    parser.add_argument("--mode", type=str, choices=["gloss_model", "precomputed"], 
                       required=True, help="Evaluation mode: use gloss_model or precomputed embeddings")
    
    # For precomputed mode
    parser.add_argument("--gloss_embeddings_path", type=str,
                       help="Path to precomputed gloss embeddings (required for precomputed mode)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "precomputed" and not args.gloss_embeddings_path:
        parser.error("--gloss_embeddings_path is required when mode is 'precomputed'")
    
    return args

        
class WSD_ViConDataset(Dataset):
    def __init__(self, samples, tokenizer, mode="precomputed", gloss_emd=None, val_mode=False):
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.span_extractor = SpanExtractor(tokenizer)
        self.mode = mode
        self.gloss_emd = gloss_emd  # Only used in precomputed mode
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

        # Compute span indices
        self.span_indices = []
        for s in tqdm(self.all_samples, desc="Computing spans", ascii=True):
            idxs = self.span_extractor.get_span_indices(
                s["sentence"], s["target_word"]
            )
            self.span_indices.append(idxs or (0, 0))

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

        if self.val_mode:
            candidate_samples = self.word2gloss[s["target_word"]]
            # Get unique gloss_id
            seen = set()
            unique_candidates = []
            for c in candidate_samples:
                if c["gloss_id"] not in seen:
                    seen.add(c["gloss_id"])
                    unique_candidates.append(c)

            candidate_ids = [c["gloss_id"] for c in unique_candidates]
            item["candidate_gloss_ids"] = candidate_ids
            
            if self.mode == "precomputed":
                # Use precomputed embeddings
                candidate_vectors = [self.gloss_emd[cid] for cid in candidate_ids]
                item["candidate_gloss_vectors"] = candidate_vectors
            else:
                # Store gloss texts for on-the-fly encoding
                candidate_glosses = [c["gloss"] for c in unique_candidates]
                item["candidate_glosses"] = candidate_glosses

        return item

    def collate_fn(self, batch):
        sentences = [b["sentence"] for b in batch]  
        spans = [b["target_spans"] for b in batch]  
        synset_ids = torch.tensor([b["synset_id"] for b in batch], dtype=torch.long)
        word_id = torch.tensor([b["word_id"] for b in batch], dtype=torch.long)
        target_words = [b["target_word"] for b in batch]
        gold_glosses = [b["gloss"] for b in batch]
        gloss_id = torch.tensor([b["gloss_id"] for b in batch], dtype=torch.long)

        result = {
            "sentence": sentences,
            "target_spans": torch.tensor(spans, dtype=torch.long),
            "synset_ids": synset_ids,
            "word_id": word_id,
            "target_words": target_words,
            "gloss": gold_glosses,
            "gloss_id": gloss_id,
        }

        if self.val_mode:
            if self.mode == "precomputed":
                # Precomputed embeddings mode
                candidate_vectors_batch = []
                candidate_ids_batch = []
                for b in batch:
                    candidate_vectors_batch.append(torch.stack(b["candidate_gloss_vectors"]))
                    candidate_ids_batch.append(torch.tensor(b["candidate_gloss_ids"], dtype=torch.long))
                
                result["candidate_gloss_vectors"] = candidate_vectors_batch
                result["candidate_gloss_ids"] = candidate_ids_batch
            else:
                # Gloss model mode
                candidate_glosses_batch = []
                candidate_ids_batch = []
                for b in batch:
                    candidate_glosses_batch.append(b["candidate_glosses"])
                    candidate_ids_batch.append(torch.tensor(b["candidate_gloss_ids"], dtype=torch.long))
                
                result["candidate_glosses"] = candidate_glosses_batch
                result["candidate_gloss_ids"] = candidate_ids_batch

        return result


def evaluate_model_precomputed(context_model, data_loader, device):
    """Evaluation using precomputed gloss embeddings"""
    valid_loss = 0.0
    TP, FP, FN = 0, 0, 0
    valid_steps = 0

    context_model.eval()
    with torch.no_grad():
        val_pbar = tqdm(data_loader, desc="Validating (Precomputed)", ascii=True)
        
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
                    "input_ids": c_toks["input_ids"],
                    "attention_mask": c_toks["attention_mask"],
                },
                target_spans
            )  # [B, H]

            for i in range(len(batch["sentence"])):
                gold_gloss_id = batch["gloss_id"][i].item()
                rF_g = batch["candidate_gloss_vectors"][i].to(device)  # [N_candidates, H]

                # Compute similarity: context_i vs N candidate glosses
                sim = torch.matmul(
                    rF_wt[i].flatten().unsqueeze(0),  # [1, H]
                    rF_g.reshape(rF_g.size(0), -1).T  # [H, N_candidates]
                )  # [1, N_candidates]

                P = F.softmax(sim, dim=1)  # [1, N_candidates]

                candidate_ids = batch["candidate_gloss_ids"][i].to(device)  # [N_candidates]
                
                # Find gold index
                gold_idx_tensor = (candidate_ids == gold_gloss_id).nonzero(as_tuple=True)[0]
                if len(gold_idx_tensor) == 0:
                    # Skip if gold gloss not found in candidates
                    continue
                gold_idx = gold_idx_tensor[0].item()

                # Compute loss
                loss_i = -torch.log(P[0, gold_idx] + 1e-8)
                valid_loss += loss_i.item()
                valid_steps += 1

                # Prediction
                pred_idx = P.argmax(dim=1).item()
                if pred_idx == gold_idx:
                    TP += 1
                else:
                    FP += 1
                    FN += 1

    # Compute metrics
    valid_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    valid_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0.0
    avg_valid_loss = valid_loss / max(1, valid_steps)

    return {
        "loss": avg_valid_loss,
        "valid_f1": valid_f1,
        "valid_recall": valid_recall,
        "valid_precision": valid_precision
    }


def evaluate_model_with_gloss_encoder(model, data_loader, device):
    """Evaluation using gloss encoder model"""
    valid_loss = 0.0
    TP, FP, FN = 0, 0, 0
    valid_steps = 0

    model.eval()
    
    with torch.no_grad():
        val_pbar = tqdm(data_loader, desc="Validating (Gloss Model)", ascii=True)
        
        for batch in val_pbar:
            # Encode contexts
            c_toks = model.tokenizer(
                batch["sentence"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            ).to(device)

            target_spans = batch["target_spans"].to(device)
            rF_wt = model.forward_context( 
                c_toks["input_ids"],
                c_toks["attention_mask"],
                target_spans
            )  # [B, H]

            for i in range(len(batch["sentence"])):
                gold_gloss_id = batch["gloss_id"][i].item()
                candidate_glosses = batch["candidate_glosses"][i]  # List of gloss texts
                candidate_ids = batch["candidate_gloss_ids"][i].to(device)  # [N_candidates]

                # Encode candidate glosses on-the-fly
                g_toks = model.tokenizer(
                    candidate_glosses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(device)

                # Get gloss embeddings
                rF_g = model.forward_gloss(
                    g_toks["input_ids"], 
                    g_toks["attention_mask"]
                )  # [N_candidates, H]

                # Compute similarity
                sim = torch.matmul(
                    rF_wt[i].flatten().unsqueeze(0),  # [1, H]
                    rF_g.reshape(rF_g.size(0), -1).T  # [H, N_candidates]
                )  # [1, N_candidates]

                P = F.softmax(sim, dim=1)  # [1, N_candidates]

                # Find gold index
                gold_idx_tensor = (candidate_ids == gold_gloss_id).nonzero(as_tuple=True)[0]
                if len(gold_idx_tensor) == 0:
                    continue
                gold_idx = gold_idx_tensor[0].item()

                # Compute loss
                loss_i = -torch.log(P[0, gold_idx] + 1e-8)
                valid_loss += loss_i.item()
                valid_steps += 1

                # Prediction
                pred_idx = P.argmax(dim=1).item()
                if pred_idx == gold_idx:
                    TP += 1
                else:
                    FP += 1
                    FN += 1

    # Compute metrics
    valid_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    valid_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0.0
    avg_valid_loss = valid_loss / max(1, valid_steps)

    return {
        "loss": avg_valid_loss,
        "valid_f1": valid_f1,
        "valid_recall": valid_recall,
        "valid_precision": valid_precision
    }


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(42) 
    args = setup_args()
    
    config = load_config("configs/poly.yml")
    
    # Load validation data
    with open(config["data"]["valid_path"], "r", encoding="utf-8") as f:
        valid_sample = json.load(f)
    
    print(f"Evaluation mode: {args.mode}")
    print(f"Context model path: {args.model_path}")
    
    # Load context model
    if args.model_type == 'vicon':
        context_model = ViSynoSenseEmbedding.from_pretrained(args.model_path).to(device)
    elif args.model_type == 'poly':
        context_model = PolyBERT.from_pretrained(args.model_path).to(device)
    elif args.model_type == 'bem':
        context_model = BiEncoderModel.from_pretrained(args.model_path).to(device)
    tokenizer = context_model.tokenizer
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Initialize based on mode
    if args.mode == "precomputed":
        print(f"Loading precomputed embeddings from: {args.gloss_embeddings_path}")
        gloss_emd = torch.load(args.gloss_embeddings_path, map_location='cpu')  # {gloss_id: tensor}
        valid_set = WSD_ViConDataset(valid_sample, tokenizer, mode="precomputed", 
                                   gloss_emd=gloss_emd, val_mode=True)
        gloss_model = None
    else:  # gloss_model mode
        valid_set = WSD_ViConDataset(valid_sample, tokenizer, mode="gloss_model", val_mode=True)
        gloss_emd = None
    
    # Create dataloader
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_set.collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    print(f"\nValidation dataset size: {len(valid_set)}")
    print(f"Number of batches: {len(valid_dataloader)}")
    print("\nStarting evaluation...")
    
    # Run evaluation based on mode
    if args.mode == "precomputed":
        valid_metrics = evaluate_model_precomputed(context_model, valid_dataloader, device)
    else:
        valid_metrics = evaluate_model_with_gloss_encoder(context_model, valid_dataloader, device)
    
    print("\n" + "="*50)
    print("  VALIDATION METRICS:")
    print("="*50)
    print(f"    Loss: {valid_metrics['loss']:.4f}")
    print(f"    F1: {valid_metrics['valid_f1']:.4f}")
    print(f"    Recall: {valid_metrics['valid_recall']:.4f}")
    print(f"    Precision: {valid_metrics['valid_precision']:.4f}")
    print("="*50)