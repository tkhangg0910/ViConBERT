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


def train_model(
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
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    if grad_clip:
        grad_clipper = AdaptiveGradientClipper(initial_max_norm=1.0)  # Lower initial norm

    history = {
        "train_loss": [],
        "valid_loss": [],
        "valid_metrics": [],
        "epoch_times": []
    }

    best_valid_f1 = 0.0
    patience_counter = 0
    global_step = 0

    model.to(device)
    print(f"[train] run_dir: {run_dir} | device: {device} | AMP: {use_amp}")

    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        model.train()
        
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_data_loader, 
                   desc=f"Training Epoch {epoch+1}/{num_epochs}",
                   ascii=True)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            global_step += 1

            context_inputs = {
                "input_ids": batch["context_input_ids"].to(device),
                "attention_mask": batch["context_attn_mask"].to(device)
            }
            target_spans = batch["target_spans"].to(device)
            candidate_glosses = batch["candidate_glosses"]
            gold_indices = batch["gold_indices"].to(device)  # FIXED: Use proper gold indices

            # Flatten all candidate glosses for batch tokenization
            all_glosses = []
            gloss_offsets = []
            start = 0
            
            for candidates in candidate_glosses:
                all_glosses.extend(candidates)
                gloss_offsets.append((start, start + len(candidates)))
                start += len(candidates)

            # Tokenize all glosses at once (more efficient)
            gloss_toks = model.tokenizer(
                all_glosses,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256
            ).to(device)

            with autocast(device_type=device):
                # Get context embeddings
                rF_wt = model.forward_context(
                    context_inputs["input_ids"],
                    context_inputs["attention_mask"], 
                    target_spans
                )  # [B, hidden_dim]

                # Get gloss embeddings
                rF_g = model.forward_gloss(
                    gloss_toks["input_ids"], 
                    gloss_toks["attention_mask"]
                )  # [total_glosses, hidden_dim]

                # Compute similarities for each instance
                batch_similarities = []
                max_candidates = max(len(candidates) for candidates in candidate_glosses)
                
                for i, (start_idx, end_idx) in enumerate(gloss_offsets):
                    ctx_emb = rF_wt[i].flatten()  # [hidden_dim]
                    gloss_embs = rF_g[start_idx:end_idx]  # [num_candidates, hidden_dim]
                    gloss_embs_flat = gloss_embs.reshape(gloss_embs.size(0), -1)
                    
                    # Compute similarity scores
                    ctx_emb = F.normalize(ctx_emb.unsqueeze(0), p=2, dim=1)
                    gloss_embs_flat = F.normalize(rF_g, p=2, dim=1)
                    # Compute similarity scores
                    similarities = torch.matmul(ctx_emb, gloss_embs_flat.T) 
                    
                    # Pad to max_candidates for batching
                    if similarities.size(1) < max_candidates:
                        pad_size = max_candidates - similarities.size(1)
                        similarities = F.pad(similarities, (0, pad_size), value=-1e9)
                    
                    batch_similarities.append(similarities)

                # Stack similarities: [B, max_candidates]
                similarities_batch = torch.cat(batch_similarities, dim=0)
                
                # Compute loss
                loss = F.cross_entropy(similarities_batch, gold_indices)
                loss = loss / grad_accum_steps

            # Backward pass
            scaler.scale(loss).backward()

            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if grad_clip:
                    grad_clipper.clip(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()

            running_loss += loss.item() * grad_accum_steps
            
            # Calculate accuracy
            preds = similarities_batch.argmax(dim=1)
            train_correct += (preds == gold_indices).sum().item()
            train_total += len(gold_indices)

            if global_step % metric_log_interval == 0:
                current_acc = train_correct / max(train_total, 1)
                pbar.set_postfix({
                    "Loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "Acc": f"{current_acc:.4f}"
                })
            else:
                pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

        # Training metrics
        train_accuracy = train_correct / max(train_total, 1)
        avg_train_loss = running_loss / len(train_data_loader)
        history["train_loss"].append(avg_train_loss)

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0

        with torch.no_grad():
            val_pbar = tqdm(valid_data_loader, desc=f"Validating {epoch+1}/{num_epochs}", ascii=True)
            
            for batch in val_pbar:
                context_inputs = {
                    "input_ids": batch["context_input_ids"].to(device),
                    "attention_mask": batch["context_attn_mask"].to(device)
                }
                target_spans = batch["target_spans"].to(device)
                candidate_glosses = batch["candidate_glosses"]
                gold_indices = batch["gold_indices"].to(device)

                # Same processing as training
                all_glosses = []
                gloss_offsets = []
                start = 0
                
                for candidates in candidate_glosses:
                    all_glosses.extend(candidates)
                    gloss_offsets.append((start, start + len(candidates)))
                    start += len(candidates)

                gloss_toks = model.tokenizer(
                    all_glosses,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256
                ).to(device)

                rF_wt = model.forward_context(
                    context_inputs["input_ids"],
                    context_inputs["attention_mask"], 
                    target_spans
                )

                rF_g = model.forward_gloss(
                    gloss_toks["input_ids"], 
                    gloss_toks["attention_mask"]
                )

                batch_similarities = []
                max_candidates = max(len(candidates) for candidates in candidate_glosses)
                
                for i, (start_idx, end_idx) in enumerate(gloss_offsets):
                    ctx_emb = rF_wt[i].flatten()
                    gloss_embs = rF_g[start_idx:end_idx]
                    gloss_embs_flat = gloss_embs.reshape(gloss_embs.size(0), -1)
                    
                    ctx_emb = F.normalize(ctx_emb.unsqueeze(0), p=2, dim=1)
                    gloss_embs_flat = F.normalize(rF_g, p=2, dim=1)
                    # Compute similarity scores
                    similarities = torch.matmul(ctx_emb, gloss_embs_flat.T) 
                    
                    if similarities.size(1) < max_candidates:
                        pad_size = max_candidates - similarities.size(1)
                        similarities = F.pad(similarities, (0, pad_size), value=-1e9)
                    
                    batch_similarities.append(similarities)

                similarities_batch = torch.cat(batch_similarities, dim=0)
                
                loss = F.cross_entropy(similarities_batch, gold_indices)
                valid_loss += loss.item()
                
                preds = similarities_batch.argmax(dim=1)
                valid_correct += (preds == gold_indices).sum().item()
                valid_total += len(gold_indices)

        valid_accuracy = valid_correct / max(valid_total, 1)
        avg_valid_loss = valid_loss / len(valid_data_loader)
        history["valid_loss"].append(avg_valid_loss)
        history["valid_metrics"].append({"accuracy": valid_accuracy})

        epoch_time = (datetime.now() - epoch_start).total_seconds()
        history["epoch_times"].append(epoch_time)

        print()
        print("=" * 80)
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Train Acc: {train_accuracy:.4f} | Time: {epoch_time:.1f}s")
        print(f"[Epoch {epoch+1}] Valid Loss: {avg_valid_loss:.4f} | Valid Acc: {valid_accuracy:.4f}")
        print("=" * 80)

        # Early stopping based on validation accuracy
        if valid_accuracy > best_valid_f1:
            best_valid_f1 = valid_accuracy
            patience_counter = 0
            
            best_model_dir = os.path.join(run_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            
            # Save model using HuggingFace method
            model.save_pretrained(best_model_dir)
            
            # Save training state
            torch.save({
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_valid_f1': best_valid_f1,
                'global_step': global_step
            }, os.path.join(best_model_dir, "training_state.pt"))
            
            patience_counter = 0

            print(f"[Best] New best valid accuracy: {best_valid_f1:.4f}")
        else:
            patience_counter += 1
            print(f"[Patience] {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            print("[EarlyStopping] Stopping training.")
            break

        # Regular checkpointing
        if (epoch + 1) % ckpt_interval == 0:
            final_model_dir = os.path.join(run_dir, "final_model")
            os.makedirs(final_model_dir, exist_ok=True)

            # Save model using HuggingFace method
            model.save_pretrained(final_model_dir)

            # Save training state
            torch.save({
                'epoch': epoch + 1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_valid_f1': best_valid_f1,
                'global_step': global_step
            }, os.path.join(final_model_dir, "training_state.pt"))

            print(f"[Checkpoint] Saved to final_model")

    return history
