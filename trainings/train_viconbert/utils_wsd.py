import os
import torch
from tqdm import tqdm
import numpy as np
from torch.amp import GradScaler, autocast
from datetime import datetime
from utils.metrics import compute_step_metrics
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


def train_model(num_epochs, train_data_loader, valid_data_loader, 
                loss_fn, optimizer, model, device, 
                checkpoint_dir,
                scheduler=None,
                early_stopping_patience=3,
                ckpt_interval=10,
                metric_log_interval=500,
                grad_clip = False,
                grad_accum_steps=1  
                ):
    """
    Train a WSD model with early stopping and checkpoint saving
    
    Args:
        num_epochs: Number of training epochs
        train_data_loader: DataLoader for training data (with weighted sampling)
        valid_data_loader: DataLoader for validation data (no shuffle)
        loss_fn: Loss function
        optimizer: Optimizer
        model: Model to train
        device: Device to train on ('cuda' or 'cpu')
        checkpoint_dir: Directory to save checkpoints
        scheduler: Learning rate scheduler (optional)
        early_stopping_patience: Number of epochs to wait before early stopping
        ckpt_interval: Interval to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    if grad_clip:
        grad_clipper = AdaptiveGradientClipper(initial_max_norm=2.0)
    scaler = GradScaler()
    history = {
        'train_loss': [],
        'train_metrics': [],
        'valid_loss': [],
        'epoch_times': [],
        'step_metrics': []
    }
    
    best_valid_loss = float('inf')
    patience_counter = 0
    global_step = 0  
    
    print(f"Training started. Checkpoints will be saved to: {run_dir}")
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        
        # ======================
        # TRAINING PHASE
        # ======================
        model.train()
        running_loss = 0.0
        train_metrics_accum = {f'recall': 0.0 }
        train_metrics_accum.update({f'precision': 0.0})

        # Training loop
        train_pbar = tqdm(train_data_loader, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}",
                         position=0, leave=True, ascii=True)
        optimizer.zero_grad() 
        current_norm = float('nan')
        for batch_idx, batch in enumerate(train_pbar):
            global_step += 1
            
            gloss_embd = batch["gloss_embd"].to(device)
            context_input_ids=batch["context_input_ids"].to(device)
            context_attention_mask=batch["context_attn_mask"].to(device)
            target_spans = None
            if "target_spans" in batch and batch["target_spans"] is not None:
                target_spans = batch["target_spans"].to(device)
            synset_ids=batch["synset_ids"].to(device)

            # optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(
                    {
                    "attention_mask":context_attention_mask,
                    "input_ids":context_input_ids
                    },
                    target_span=target_spans

                )

                loss = loss_fn(outputs,gloss_embd, synset_ids)
                loss = loss / grad_accum_steps
            scaler.scale(loss).backward()
            if (batch_idx + 1) % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                if grad_clip:
                    current_norm = grad_clipper.clip(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Calculate training metrics
            running_loss += loss.item()
            

            batch_metrics = compute_step_metrics(outputs, synset_ids, 
                                               k_vals=(1,), 
                                               device=device, ndcg=False)
            

            train_metrics_accum[f'recall']    += batch_metrics[f'recall@1']
            train_metrics_accum[f'precision'] += batch_metrics[f'precision@1']


            # Update progress bar
            if global_step % metric_log_interval == 0:
                step_metrics = batch_metrics.copy()
                step_metrics['step'] = global_step
                step_metrics['loss'] = loss.item()
                history['step_metrics'].append(step_metrics)
                
            else:
                postfix = {
                    'Loss': f'{loss.item():.4f}',
                }
                if grad_clip:
                    postfix["Grad"] = f'{current_norm:.2f}'
                    postfix["Clip"] = f'{grad_clipper.max_norm:.2f}'

                train_pbar.set_postfix(postfix)
            
            # del outputs, loss, gloss_embd, context_input_ids, context_attention_mask, target_spans, synset_ids
            del outputs, loss, gloss_embd, context_input_ids, context_attention_mask, synset_ids
            if target_spans is not None:
                del target_spans
            torch.cuda.empty_cache()  
            # if scheduler:
            #     scheduler.step() 
                
        train_metrics = {}
        num_batches = len(train_data_loader)
        train_metrics = {f'recall': train_metrics_accum[f'recall'] / num_batches}
        train_metrics.update({f'precision': train_metrics_accum[f'precision'] / num_batches})

        epsilon = 1e-10
        
        precision = train_metrics[f'precision']
        recall = train_metrics[f'recall']
        f1 = 2 * precision * recall / (precision + recall + epsilon)
        train_metrics[f'f1'] = f1

        train_loss = running_loss / num_batches
        train_metrics['loss'] = train_loss
        history['train_metrics'].append(train_metrics)
        
        # ======================
        # VALIDATION PHASE
        # ======================
        print(f"\nValidating epoch {epoch+1}...")
        valid_metrics = evaluate_model(model, valid_data_loader, device)
        
        # ======================
        # SCHEDULER STEP (EPOCH-LEVEL)
        # ======================
        if scheduler:
            scheduler.step(valid_metrics['loss'])
            
        # Calculate final training metrics
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        train_loss = running_loss / len(train_data_loader)

        # Update history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_metrics['loss'])
        history['epoch_times'].append(epoch_time)

        # ======================
        # CHECKPOINTING
        # ======================
        if (epoch + 1) % ckpt_interval == 0:
            checkpoint_dir_path  = os.path.join(run_dir, f"epoch_{epoch+1}")
            os.makedirs(checkpoint_dir_path, exist_ok=True)

            model.save_pretrained(checkpoint_dir_path)

            training_state = {
                'epoch': epoch+1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
                'global_step': global_step
            }
            torch.save(training_state, os.path.join(checkpoint_dir_path, "training_state.pt"))
            print(f"Checkpoint saved to directory: {checkpoint_dir_path}")

            
        # ======================
        # EARLY STOPPING & BEST MODEL SAVING
        # ======================
        if valid_metrics['loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['loss']
            best_model_dir = os.path.join(run_dir, "best_model")
            os.makedirs(best_model_dir, exist_ok=True)
            
            # Save model using new method
            model.save_pretrained(best_model_dir)
            
            # Save training state
            torch.save({
                'epoch': epoch+1,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_valid_loss': best_valid_loss,
                'global_step': global_step
            }, os.path.join(best_model_dir, "training_state.pt"))
            
            patience_counter = 0
            print(f"✓ New best model saved to: {best_model_dir}")
        else:
            patience_counter += 1
        
        # ======================
        # REGENERATE BATCHES FOR NEXT EPOCH
        # ======================
        if hasattr(train_data_loader.dataset, 'on_epoch_end'):
            print("Regenerating batches for next epoch...")
            train_data_loader.dataset.on_epoch_end()
            
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"  Time: {epoch_time:.2f}s")

        # Training metrics
        print("\n  TRAIN METRICS (avg per batch):")
        print(f"    Loss: {train_loss:.4f}")
        line = (
            f"    Recall:     {train_metrics[f'recall']:.4f} | "
            f"Precision:  {train_metrics[f'precision']:.4f} | "
            f"F1:         {train_metrics[f'f1']:.4f} | "
        )

        print(line)


        # Validation metrics
        print("\n  VALIDATION METRICS:")
        print(f"    Loss: {valid_metrics['loss']:.4f}")
        print(
            f"    Recall:     {valid_metrics[f'valid_accuracy']:.4f} | "
            f"Precision:  {valid_metrics[f'valid_accuracy']:.4f} | "
            f"F1:         {valid_metrics[f'valid_accuracy']:.4f} | "
        )

        
        print(f"\n  Early stopping: {patience_counter}/{early_stopping_patience}")
        
        if scheduler and hasattr(scheduler, 'get_last_lr'):
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        print(f"{'='*60}")
        
        # Early stopping check
        if patience_counter >= early_stopping_patience:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs!")
            print(f"Best validation loss: {best_valid_loss:.4f}")
            break

    # ======================
    # FINAL SAVE
    # ======================
    final_model_dir = os.path.join(run_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)

    # Save model using new method
    model.save_pretrained(final_model_dir)

    # Save training state
    torch.save({
        'epoch': epoch+1,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_valid_loss': best_valid_loss,
        'global_step': global_step
    }, os.path.join(final_model_dir, "training_state.pt"))
    
    history_path = os.path.join(run_dir, "training_history.pt")
    torch.save(history, history_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Best validation loss: {best_valid_loss:.4f}")
    print(f"💾 All files saved to: {run_dir}")
    
    return history, model

import torch.nn.functional as F

import torch
import torch.nn.functional as F
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    """Enhanced evaluation function compatible with WSD_ViConDataset"""
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    
    with torch.inference_mode():
        eval_pbar = tqdm(data_loader, desc="Evaluating", position=0, leave=False, ascii=True)
        
        for batch in eval_pbar:
            # Extract data from batch (matching your dataset structure)
            context_input_ids = batch["context_input_ids"].to(device)
            context_attn_mask = batch["context_attn_mask"].to(device)
            target_spans = batch["target_spans"].to(device) if batch["target_spans"] is not None else None
            synset_ids = batch["synset_ids"].to(device)
            
            candidate_gloss_vectors = batch["candidate_gloss_vectors"]  # List of tensors
            candidate_gloss_ids = batch["candidate_gloss_ids"]  # List of tensors
            gloss_ids = batch["gloss_id"]  # Gold gloss IDs
            
            # Get context representations
            context_inputs = {
                "input_ids": context_input_ids,
                "attention_mask": context_attn_mask
            }
            
            outputs = model(context_inputs, target_span=target_spans)
            
            batch_similarities = []
            batch_gold_indices = []
            
            for i in range(len(candidate_gloss_vectors)):
                # Context embedding for sample i
                ctx_emb = outputs[i].unsqueeze(0)  # (1, hidden_dim)
                
                # Candidate gloss vectors for sample i
                candidate_vectors = candidate_gloss_vectors[i].to(device)  # (num_candidates, hidden_dim)
                candidate_ids = candidate_gloss_ids[i]  # (num_candidates,)
                gold_gloss_id = gloss_ids[i].item()
                
                # Find gold index in candidates
                gold_idx = None
                for j, cand_id in enumerate(candidate_ids):
                    if cand_id.item() == gold_gloss_id:
                        gold_idx = j
                        break
                
                if gold_idx is None:
                    # Skip this sample if gold is not in candidates
                    continue
                
                # Normalize embeddings
                ctx_emb_norm = F.normalize(ctx_emb, p=2, dim=1)
                candidate_vectors_norm = F.normalize(candidate_vectors, p=2, dim=1)
                
                # Compute similarities
                similarities = torch.matmul(ctx_emb_norm, candidate_vectors_norm.T)  # (1, num_candidates)
                
                batch_similarities.append(similarities.squeeze(0))  # (num_candidates,)
                batch_gold_indices.append(gold_idx)
            
            if batch_similarities:
                # Pad similarities to same length for batching
                max_candidates = max(sim.size(0) for sim in batch_similarities)
                padded_similarities = []
                
                for sim in batch_similarities:
                    if sim.size(0) < max_candidates:
                        pad_size = max_candidates - sim.size(0)
                        sim = F.pad(sim.unsqueeze(0), (0, pad_size), value=-1e9).squeeze(0)
                    padded_similarities.append(sim)
                
                similarities_tensor = torch.stack(padded_similarities)  # (batch_size, max_candidates)
                gold_indices_tensor = torch.tensor(batch_gold_indices, dtype=torch.long, device=device)
                
                # Calculate loss
                loss = F.cross_entropy(similarities_tensor, gold_indices_tensor)
                valid_loss += loss.item()
                
                # Calculate accuracy
                preds = similarities_tensor.argmax(dim=1)
                valid_correct += (preds == gold_indices_tensor).sum().item()
                valid_total += len(gold_indices_tensor)
            
            eval_pbar.set_postfix({'Loss': f'{valid_loss/(eval_pbar.n+1):.4f}'})

    if valid_total == 0:
        print("Warning: No valid samples found during evaluation")
        return {
            'loss': float('inf'),
            'valid_accuracy': 0.0
        }

    valid_accuracy = valid_correct / valid_total
    avg_valid_loss = valid_loss / len(data_loader)

    return {
        'loss': avg_valid_loss,
        'valid_accuracy': valid_accuracy
    }


def evaluate_model_with_loss_fn(model, data_loader, loss_fn, device):
    """Alternative evaluation function that uses the same loss function as training"""
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    valid_total = 0
    
    with torch.inference_mode():
        eval_pbar = tqdm(data_loader, desc="Evaluating", position=0, leave=False, ascii=True)
        
        for batch in eval_pbar:
            context_input_ids = batch["context_input_ids"].to(device)
            context_attn_mask = batch["context_attn_mask"].to(device)
            target_spans = batch["target_spans"].to(device) if batch["target_spans"] is not None else None
            gloss_embd = batch["gloss_embd"].to(device)
            synset_ids = batch["synset_ids"].to(device)
            
            context_inputs = {
                "input_ids": context_input_ids,
                "attention_mask": context_attn_mask
            }
            
            outputs = model(context_inputs, target_span=target_spans)
            
            # Use the same loss function as training
            loss = loss_fn(outputs, gloss_embd, synset_ids)
            valid_loss += loss.item()
            
            # Calculate accuracy based on cosine similarity
            outputs_norm = F.normalize(outputs, p=2, dim=1)
            gloss_embd_norm = F.normalize(gloss_embd, p=2, dim=1)
            similarities = torch.sum(outputs_norm * gloss_embd_norm, dim=1)
            
            # You can adjust this threshold based on your task
            threshold = 0.5
            correct_predictions = (similarities > threshold).sum().item()
            valid_correct += correct_predictions
            valid_total += len(similarities)
            
            eval_pbar.set_postfix({'Loss': f'{valid_loss/(eval_pbar.n+1):.4f}'})

    valid_accuracy = valid_correct / max(valid_total, 1)
    avg_valid_loss = valid_loss / len(data_loader)

    return {
        'loss': avg_valid_loss,
        'valid_accuracy': valid_accuracy
    }