import argparse
import json
import os
import torch
from transformers import PhobertTokenizerFast, XLMRobertaTokenizerFast
from torch.utils.data import DataLoader
from data.processed.stage1_pseudo_sents.pseudo_sent_datasets import PseudoSents_Dataset, PseudoSentsFlatDataset
from utils.load_config import load_config
from models.base_model import ViSynoSenseEmbedding
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
    parser.add_argument("--model_type", type=str, help="Model path")
    parser.add_argument("--batch_size", type=int,default=768, help="Batch size")
    args = parser.parse_args()
    return args 

def evaluate_model(model, data_loader, loss_fn, device, metric_k_vals=(1, 5, 10)):
    """Enhanced evaluation with detailed metrics"""
    model.eval()
    running_loss = 0.0
    
    all_embeddings = []
    all_labels = []
    
    with torch.inference_mode():
        eval_pbar = tqdm(data_loader, desc="Evaluating", position=0, leave=False,ascii=True)
        for batch in eval_pbar:
            gloss_embd = batch["gloss_embd"].to(device)
            context_input_ids=batch["context_input_ids"].to(device)
            context_attention_mask=batch["context_attn_mask"].to(device)
            target_spans = None
            if "target_spans" in batch and batch["target_spans"] is not None:
                target_spans = batch["target_spans"].to(device)
            synset_ids=batch["synset_ids"].to(device)
            
            with autocast(device_type=device):
                outputs = model(
                    {
                    "attention_mask":context_attention_mask,
                    "input_ids":context_input_ids
                    },
                    target_span=target_spans
                )
                
                loss = loss_fn(outputs, gloss_embd, synset_ids)
            
            
            running_loss += loss.item()
            
            all_embeddings.append(outputs)
            all_labels.append(synset_ids)

    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = running_loss / len(data_loader)
        
    hard  = compute_full_metrics_large_scale(
        all_embeddings, 
        all_labels, 
        k_vals=metric_k_vals, 
        device=device
    )
    N, D = all_embeddings.shape
    if device.startswith('cuda'):
        total_mem = torch.cuda.get_device_properties(device).total_memory
        used_mem  = torch.cuda.memory_allocated(device)
        free_mem  = total_mem - used_mem
        # mỗi row sim uses 4 bytes * D entries
        mem_per_row = 4 * D * N
        chunk_size = max(1, int(free_mem / mem_per_row))
        chunk_size = min(chunk_size, 5000)  # giới hạn trên nếu cần
    else:
        chunk_size = 1000

        # label → raw synset_id
    label_to_synset_map = data_loader.dataset.global_label_to_synset 

    ndcg = compute_ndcg_from_faiss(
        context_embd=all_embeddings,
        true_synset_labels=all_labels,
        faiss_index_path="data/processed/gloss_faiss.index",
        synset_id_map_path="data/processed/synset_ids.pt",
        label_to_synset_map=label_to_synset_map,
        k_vals=(1, 5, 10)
    )

    
    return {
        'loss': avg_loss,
        **hard,
         **ndcg
    }

if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(42) 
    args = setup_args()
    
    config = load_config(f"configs/{args.model_type}.yml")
    
    with open(config["data"]["valid_path"], "r",encoding="utf-8") as f:
        valid_sample = json.load(f)
        
    tokenizer = PhobertTokenizerFast.from_pretrained(args.model_path)
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
    valid_set = PseudoSentsFlatDataset(config["data"]["emd_path"],
                                # gloss_enc,
                                valid_sample, tokenizer)
    
    valid_dataloader = DataLoader(
        valid_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=valid_set.collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    model = ViSynoSenseEmbedding.from_pretrained(args.model_path).to(device)
    loss_fn = InfonceDistillLoss(aux_weight=1)

    metric_k_vals=(1, 5, 10)
    print("\nValidating epoch...")
    valid_metrics = evaluate_model(model, valid_dataloader, loss_fn, device, metric_k_vals)
    
    print("\n  VALIDATION METRICS:")
    print(f"    Loss: {valid_metrics['loss']:.4f}")
    for k in metric_k_vals:
        print(
            f"    Recall@{k}:     {valid_metrics[f'recall@{k}']:.4f} | "
            f"Precision@{k}:  {valid_metrics[f'precision@{k}']:.4f} | "
            f"F1@{k}:         {valid_metrics[f'f1@{k}']:.4f} | "
            f"ndcg@{k}:      {valid_metrics[f'ndcg@{k}']:.4f} | "
        )