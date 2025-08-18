import os
import torch
from tqdm import tqdm
import numpy as np
from torch.amp import GradScaler, autocast
from datetime import datetime
import torch.nn.functional as F
from utils.metrics import compute_precision_recall_f1_for_wsd

def log_gpu_memory(tag=""):
    allocated = torch.cuda.memory_allocated() / 1024**2  # MB
    reserved  = torch.cuda.memory_reserved() / 1024**2   # MB
    print(f"[GPU Memory] {tag} | Allocated: {allocated:.1f} MB | Reserved: {reserved:.1f} MB")

class AdaptiveGradientClipper:
    def __init__(self, initial_max_norm=1.0):
        self.max_norm = initial_max_norm
        self.history = []
        
    def clip(self, model):
        parameters = [p for p in model.parameters() if p.grad is not None]
        total_norm = torch.norm(torch.stack(
            [torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
        
        self.history.append(total_norm.item())
        
        if len(self.history) % 100 == 0:
            p95 = np.percentile(self.history[-100:], 95)
            if p95 > self.max_norm * 1.5:
                self.max_norm *= 1.2  
            elif p95 < self.max_norm * 0.5:
                self.max_norm *= 0.8 
        
        torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        return total_norm.item()
    
def compute_context_vs_gloss_similarity(model, context_inputs, target_idx, input_ids,attention_mask, device, chunk_size=256):
    """
    context_inputs: dict with 'input_ids' and 'attention_mask', shape [B, Lc]
    target_idx: tensor [B, 2] target spans
    gloss_list: list of all gloss strings (~5000)
    Returns: similarity matrix [B, num_gloss]
    """
    B = context_inputs["input_ids"].size(0)
    num_gloss = len(input_ids)
    
    # Encode contexts once
    rF_wt = model.forward_context(
        context_inputs["input_ids"].to(device),
        context_inputs["attention_mask"].to(device),
        target_idx.to(device)
    )  # [B, H] or [B, polym, H] depending on your model
    rF_wt_flat = rF_wt.reshape(B, -1)  # flatten if needed
    # log_gpu_memory(f"After Context")

    # Encode glosses in chunks
    sim_list = []
    for i in range(0, num_gloss, chunk_size):
        chunk_input_ids = input_ids[i:i+chunk_size].to(device)
        chunk_attention_mask = attention_mask[i:i+chunk_size].to(device)
        # log_gpu_memory(f"Before forward chunk {i//chunk_size}")


        with autocast(device_type=device):
            rF_g = model.forward_gloss(chunk_input_ids, chunk_attention_mask)  # [chunk_size, H]
            rF_g_flat = rF_g.reshape(len(chunk_input_ids), -1)
            sim_chunk = torch.matmul(rF_wt_flat, rF_g_flat.T)  # [B, chunk_size]
        sim_list.append(sim_chunk.cpu())  
        # log_gpu_memory(f"After forward chunk {i//chunk_size}")


    # 4Concatenate all chunk similarities -> [B, num_gloss]
    similarity = torch.cat(sim_list, dim=1)  # now on CPU
    similarity = similarity.to(device)       
    return similarity


def train_model_bc(
    num_epochs,
    train_data_loader,
    valid_data_loader,
    optimizer,
    model,
    device,
    checkpoint_dir,
    scheduler=None,
    early_stopping_patience=3,
    ckpt_interval=10,
    metric_log_interval=500,
    grad_clip=False,
    grad_accum_steps=1,
    loss_fn=None,               
    save_optimizer_state=True
):
    """
    Training loop adapted for PolyBERT.

    Expects each batch from train_data_loader to contain at least:
      - context_input_ids: LongTensor [B, Lc]
      - context_attn_mask:  LongTensor [B, Lc]
      - gloss_input_ids:    LongTensor [B, Lg]
      - gloss_attn_mask:    LongTensor [B, Lg]
      - target_idx:         LongTensor [B]  (position index of target token in context)
    Optionally:
      - synset_ids or labels for metric computation

    Model.forward should accept:
      model(context_inputs_dict, gloss_inputs_dict, target_idx)
    and return (loss_tensor, MF_matrix)
    """

    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    if grad_clip:
        grad_clipper = AdaptiveGradientClipper(initial_max_norm=2.0)

    history = {
        "train_loss": [],
        "valid_loss": [],
        "step_metrics": [],
        "epoch_times": [],
        "valid_metrics":[]
    }

    best_valid_loss = float("inf")
    patience_counter = 0
    global_step = 0

    model.to(device)
    
    print(f"[train] run_dir: {run_dir}  | device: {device} | AMP: {use_amp}")

    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        model.train()
        train_tp, train_fp, train_fn = 0, 0, 0  # accumulate for epoch

        running_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_data_loader, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}",
                         position=0, leave=True, ascii=True)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            global_step += 1
            train_steps += 1
            c_toks = model.tokenizer(
                batch["sentence"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            )
            g_toks = model.tokenizer(
                batch["gloss"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
                return_attention_mask=True
            )
            context_inputs = {
                "input_ids": c_toks["context_input_ids"].to(device),
                "attention_mask": c_toks["context_attn_mask"].to(device)
            }
            gloss_inputs = {
                "input_ids": g_toks["context_input_ids"].to(device),
                "attention_mask": g_toks["context_attn_mask"].to(device)
            }
            gid = batch["gloss_id"].to(device)
            target_idx = batch["target_spans"].to(device)  # shape [B]
            # forward + loss (use AMP if available)
            with autocast(device_type=device):
                rF_wt, rF_g = model(context_inputs ,gloss_inputs,target_idx)  

                loss, sim  =model.batch_contrastive_loss(rF_wt, rF_g, gid)
                loss = loss / float(grad_accum_steps)



            scaler.scale(loss).backward()

            # gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if grad_clip:
                    current_norm = grad_clipper.clip(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                if scheduler is not None:
                    try:
                        scheduler.step()
                    except:
                        pass

            running_loss += loss.item() * float(grad_accum_steps)

            # optional metrics
            B = rF_wt.size(0)

            preds = sim.argmax(dim=1)
            correct = (preds == torch.arange(B, device=device))
            train_tp += correct.sum().item()
            train_fp += (B - correct.sum().item())
            train_fn += (B - correct.sum().item())

            if global_step % metric_log_interval == 0:
                postfix = {"Loss": f"{(running_loss / max(1, train_steps)):.4f}"}
                if grad_clip:
                    postfix["GradMax"] = f"{grad_clipper.max_norm:.2f}"
                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"loss": f"{loss.item()*grad_accum_steps:.4f}"})
        
        precision = train_tp / (train_tp + train_fp) if (train_tp + train_fp) > 0 else 0.0
        recall    = train_tp / (train_tp + train_fn) if (train_tp + train_fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        train_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        
        avg_train_loss = running_loss / max(1, train_steps)
        history["train_loss"].append(avg_train_loss)

        # ============= VALIDATION =============
        valid_loss = 0.0
        TP, FP, FN = 0, 0, 0
        valid_steps = 0

        model.eval()
        with torch.no_grad():
            val_pbar = tqdm(valid_data_loader, 
                         desc=f"Validating {epoch+1}/{num_epochs}",
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
                    gold_gloss = batch["gold_glosses"][i]

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
        valid_f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        avg_valid_loss  = valid_loss / max(1, valid_steps)

        
        history["valid_loss"].append(avg_valid_loss)
        history["valid_metrics"].append({
            "precision": valid_precision,
            "recall": valid_recall,
            "f1": valid_f1
        })


        epoch_time = (datetime.now() - epoch_start).total_seconds()
        history["epoch_times"].append(epoch_time)
        print()
        print("===============================================================================")
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | time: {epoch_time:.1f}s")
        print(f"Train Recall: {train_metrics['recall']:.4f} | Train Precision: {train_metrics['precision']:.4f} | Train F1: {train_metrics['f1']:.4f} | ")
        print()
        print(f"[Epoch {epoch+1}] Valid Loss: {avg_valid_loss:.4f} | time: {epoch_time:.1f}s")
        print(f"Valid Recall: {valid_recall:.4f} | Valid Precision: {valid_precision:.4f} | Valid F1: {valid_f1:.4f} | ")
        print("===============================================================================")
        print()
        # checkpointing
        ckpt_path = os.path.join(run_dir, f"epoch_{epoch+1}.pt")
        if (epoch + 1) % ckpt_interval == 0:
            save_obj = {"epoch": epoch + 1, "model_state": model.state_dict()}
            if save_optimizer_state:
                save_obj["optimizer_state"] = optimizer.state_dict()
            torch.save(save_obj, ckpt_path)
            # also save model.pretrained format if you want
            try:
                model.save_pretrained(os.path.join(run_dir, f"model_epoch_{epoch+1}"))
            except Exception:
                pass
            print(f"[Checkpoint] saved to {ckpt_path}")

        # early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            patience_counter = 0
            # save best model
            best_path = os.path.join(run_dir, "best_model.pt")
            save_obj = {"epoch": epoch + 1, "model_state": model.state_dict()}
            if save_optimizer_state:
                save_obj["optimizer_state"] = optimizer.state_dict()
            torch.save(save_obj, best_path)
            try:
                model.save_pretrained(os.path.join(run_dir, "best_model_pretrained"))
            except Exception:
                pass
            print(f"[Best] new best valid loss {best_valid_loss:.4f} -> saved best model.")
        else:
            patience_counter += 1
            print(f"[Patience] {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print("[EarlyStopping] stopping training.")
            break

    return history