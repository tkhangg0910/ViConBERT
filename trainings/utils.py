import os
import torch
from tqdm import tqdm
import numpy as np
from torch.amp import GradScaler, autocast
from datetime import datetime
from utils.metrics import compute_step_metrics, compute_full_metrics_large_scale, compute_ndcg_from_faiss

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
                metric_k_vals=(1, 5, 10),
                metric_log_interval=500
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
    grad_clipper = AdaptiveGradientClipper(initial_max_norm=1.0)
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
        train_metrics_accum = {f'recall@{k}': 0.0 for k in metric_k_vals}
        train_metrics_accum.update({f'precision@{k}': 0.0 for k in metric_k_vals})
        train_metrics_accum.update({f'ndcg@{k}': 0.0 for k in metric_k_vals})

        # Training loop
        train_pbar = tqdm(train_data_loader, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}",
                         position=0, leave=True, ascii=True)
        
        for batch_idx, batch in enumerate(train_pbar):
            global_step += 1
            
            gloss_embd = batch["gloss_embd"].to(device)
            context_input_ids=batch["context_input_ids"].to(device)
            context_attention_mask=batch["context_attn_mask"].to(device)
            target_spans = None
            if "target_spans" in batch and batch["target_spans"] is not None:
                target_spans = batch["target_spans"].to(device)
            synset_ids=batch["synset_ids"].to(device)

            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(
                    {
                    "attention_mask":context_attention_mask,
                    "input_ids":context_input_ids
                    },
                    target_span=target_spans

                )

                loss = loss_fn(outputs,gloss_embd, synset_ids)
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            current_norm = grad_clipper.clip(model)

            scaler.step(optimizer)
            scaler.update()
            
            # Calculate training metrics
            running_loss += loss.item()
            

            batch_metrics = compute_step_metrics(outputs, synset_ids, 
                                               k_vals=metric_k_vals, 
                                               device=device)
            

            for k in metric_k_vals:
                train_metrics_accum[f'recall@{k}']    += batch_metrics[f'recall@{k}']
                train_metrics_accum[f'precision@{k}'] += batch_metrics[f'precision@{k}']
                train_metrics_accum[f'ndcg@{k}']    += batch_metrics[f'ndcg@{k}']


            # Update progress bar
            if global_step % metric_log_interval == 0:
                step_metrics = batch_metrics.copy()
                step_metrics['step'] = global_step
                step_metrics['loss'] = loss.item()
                history['step_metrics'].append(step_metrics)
                
            else:
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Grad': f'{current_norm:.2f}',
                    'Clip': f'{grad_clipper.max_norm:.2f}'
                })

            
            # if scheduler:
            #     scheduler.step() 
                
        train_metrics = {}
        num_batches = len(train_data_loader)
        train_metrics = {f'recall@{k}': train_metrics_accum[f'recall@{k}'] / num_batches
                 for k in metric_k_vals}
        train_metrics.update({f'precision@{k}': train_metrics_accum[f'precision@{k}'] / num_batches
                            for k in metric_k_vals})
        train_metrics.update({f'ndcg@{k}': train_metrics_accum[f'ndcg@{k}'] / num_batches
                            for k in metric_k_vals})


        train_loss = running_loss / num_batches
        train_metrics['loss'] = train_loss
        history['train_metrics'].append(train_metrics)
        
        # ======================
        # VALIDATION PHASE
        # ======================
        print(f"\nValidating epoch {epoch+1}...")
        valid_metrics = evaluate_model(model, valid_data_loader, loss_fn, device, metric_k_vals)
        
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
        for k in metric_k_vals:
            print(
                f"    Recall@{k}:     {train_metrics[f'recall@{k}']:.4f} | "
                f"Precision@{k}:  {train_metrics[f'precision@{k}']:.4f} | "
                f"ndcg@{k}:      {train_metrics[f'ndcg@{k}']:.4f} | "
            )

        # Validation metrics
        print("\n  VALIDATION METRICS:")
        print(f"    Loss: {valid_metrics['loss']:.4f}")
        for k in metric_k_vals:
            print(
                f"    Recall@{k}:     {valid_metrics[f'recall@{k}']:.4f} | "
                f"Precision@{k}:  {valid_metrics[f'precision@{k}']:.4f} | "
                f"F1@{k}:         {valid_metrics[f'f1@{k}']:.4f} | "
                f"ndcg@{k}:      {valid_metrics[f'ndcg@{k}']:.4f} | "
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
