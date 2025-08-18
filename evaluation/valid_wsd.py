import argparse
import json
import os
import torch
from transformers import PhobertTokenizerFast, XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
from data.processed.polybert_exp.dataset import PolyBERTtDatasetV3
from utils.load_config import load_config
from models.polybert import PolyBERT
import torch
from tqdm import tqdm
from transformers.utils import is_torch_available

from torch.amp import autocast
from utils.metrics import compute_step_metrics, compute_full_metrics_large_scale, compute_ndcg_from_faiss


if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
from utils.loss_fn import InfonceDistillLoss
def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model_path", type=str, help="Model path")
    parser.add_argument("--batch_size", type=int,default=768, help="Batch size")
    args = parser.parse_args()
    return args 
import torch.nn.functional as F

def evaluate_model(model, data_loader, device):
    """Enhanced evaluation with detailed metrics"""
    valid_loss = 0.0
    TP, FP, FN = 0, 0, 0
    valid_steps = 0

    model.eval()
    with torch.no_grad():
        val_pbar = tqdm(data_loader, 
                        desc=f"Validating",
                        position=1, leave=True, ascii=True)
        for batch in val_pbar:
            c_toks = model.tokenizer(
                batch["sentence"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            ).to(device)


            target_spans = batch["target_spans"].to(device)
            rF_wt = model.forward_context(c_toks["input_ids"],
                                        c_toks["attention_mask"],
                                        target_spans)  # [B, polym, H]

            for i in range(len(batch)):
                candidates = batch["candidate_glosses"][i]  # list of N gloss strings
                gold_gloss = batch["gloss"][i]

                # tokenize candidate glosses
                g_toks = model.tokenizer(candidates,
                                        return_tensors="pt",
                                        padding=True,
                                        truncation=True,
                                        max_length=256).to(device)

                rF_g = model.forward_gloss(g_toks["input_ids"],
                                        g_toks["attention_mask"])  # [N, polym, H]

                # similarity context_i vs N candidate glosses
                sim = torch.matmul(rF_wt[i].flatten().unsqueeze(0), rF_g.reshape(len(candidates),-1).T)  # [1, N]
                P = F.softmax(sim, dim=1)  # [1, N]

                gold_idx = candidates.index(gold_gloss)
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

    # compute metrics
    valid_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    valid_recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    valid_f1        = 2 * valid_precision * valid_recall / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0.0
    avg_valid_loss  = valid_loss / max(1, valid_steps)


    return {
        "loss": avg_valid_loss,
        "valid_f1":valid_f1,
        "valid_recall":valid_recall,
        "valid_precision":valid_precision
    }

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(42) 
    args = setup_args()
    
    config = load_config("configs/poly.yml")
    
    with open(config["data"]["valid_path"], "r",encoding="utf-8") as f:
        valid_sample = json.load(f)
        
    tokenizer = PhobertTokenizerFast.from_pretrained(args.model_path)
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    valid_set = PolyBERTtDatasetV3(valid_sample, tokenizer, val_mode=True)
    
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_set.collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    model = PolyBERT.from_pretrained(args.model_path).to(device)
    print("\nValidating epoch...")
    valid_metrics = evaluate_model(model, valid_dataloader, device)
    
    print("\n  VALIDATION METRICS:")
    print(f"    Loss: {valid_metrics['loss']:.4f}")
    print(f"    Loss: {valid_metrics['valid_f1']:.4f}")
    print(f"    Loss: {valid_metrics['valid_recall']:.4f}")
    print(f"    Loss: {valid_metrics['valid_precision']:.4f}")

