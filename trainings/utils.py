import os
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from datetime import datetime
from utils.metrics import compute_step_metrics, compute_full_metrics_large_scale

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
        num_batches = 0  
        train_metrics_accum = {f'recall@{k}': 0.0 for k in metric_k_vals}
        train_metrics_accum.update({f'precision@{k}': 0.0 for k in metric_k_vals})

        # Training loop
        train_pbar = tqdm(train_data_loader, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}",
                         position=0, leave=True)
        
        for batch_idx, batch in enumerate(train_pbar):
            global_step += 1
            num_batches += 1
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attn_mask'].to(device)
            span_indices = batch['span_indices'].to(device)
            synset_ids = batch['synset_ids'].to(device)

            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(input_ids, attention_mask, span_indices=span_indices)
                loss = loss_fn(outputs, synset_ids)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                    # Learning rate scheduling
            
            # Calculate training metrics
            running_loss += loss.item()
            

            batch_metrics = compute_step_metrics(outputs, synset_ids, 
                                               k_vals=metric_k_vals, 
                                               device=device)

            for k in metric_k_vals:
                train_metrics_accum[f'recall@{k}'] += batch_metrics[f'recall@{k}']
                train_metrics_accum[f'precision@{k}'] += batch_metrics[f'precision@{k}']

            # Update progress bar
            if global_step % metric_log_interval == 0:
                step_metrics = batch_metrics.copy()
                step_metrics['step'] = global_step
                step_metrics['loss'] = loss.item()
                history['step_metrics'].append(step_metrics)
                
                # Hiển thị metrics trong progress bar
                train_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'R@5': f"{step_metrics['recall@5']:.4f}",
                    'P@5': f"{step_metrics['precision@5']:.4f}"
                })
            else:
                train_pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
            
            if scheduler:
                scheduler.step() 
                
        train_metrics = {}
        for k in metric_k_vals:
            train_metrics[f'recall@{k}'] = train_metrics_accum[f'recall@{k}'] / num_batches
            train_metrics[f'precision@{k}'] = train_metrics_accum[f'precision@{k}'] / num_batches
        train_loss = running_loss / num_batches
        train_metrics['loss'] = train_loss
        history['train_metrics'].append(train_metrics)

        # ======================
        # VALIDATION PHASE
        # ======================
        print(f"\nValidating epoch {epoch+1}...")
        valid_metrics = evaluate_model(model, valid_data_loader, loss_fn, device, metric_k_vals)
        
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
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_metrics': train_metrics,
                'valid_metrics': valid_metrics,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            
        # ======================
        # EARLY STOPPING & BEST MODEL SAVING
        # ======================
        if valid_metrics['loss'] < best_valid_loss:
            best_valid_loss = valid_metrics['loss']
            best_model_path = os.path.join(run_dir, "best_model.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_valid_loss': best_valid_loss,
                'history': history
            }, best_model_path)
            patience_counter = 0
            print(f"✓ New best model saved! Valid Loss: {best_valid_loss:.4f}")
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
            print(f"    Recall@{k}: {train_metrics[f'recall@{k}']:.4f} | Precision@{k}: {train_metrics[f'precision@{k}']:.4f}")
        
        # Validation metrics
        print("\n  VALIDATION METRICS:")
        print(f"    Loss: {valid_metrics['loss']:.4f}")
        for k in metric_k_vals:
            print(f"    Recall@{k}: {valid_metrics[f'recall@{k}']:.4f} | Precision@{k}: {valid_metrics[f'precision@{k}']:.4f}")
        # print(f"    NMI: {valid_metrics['nmi']:.4f}")
        
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
    final_model_path = os.path.join(run_dir, "final_model.pt")
    torch.save({
        'epoch': epoch+1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_valid_loss': best_valid_loss,
        'history': history
    }, final_model_path)
    
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
        eval_pbar = tqdm(data_loader, desc="Evaluating", position=0, leave=False)
        for batch in eval_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attn_mask'].to(device)
            span_indices = batch['span_indices'].to(device) 
            synset_ids = batch['synset_ids'].to(device)
            
            with autocast(device_type=device):
                outputs = model(input_ids, attention_mask, span_indices=span_indices)
                loss = loss_fn(outputs, synset_ids)
            
            running_loss += loss.item()
            
            all_embeddings.append(outputs)
            all_labels.append(synset_ids)

    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    avg_loss = running_loss / len(data_loader)
        
    full_metrics = compute_full_metrics_large_scale(
        all_embeddings, 
        all_labels, 
        k_vals=metric_k_vals, 
        device=device
    )

    
    return {
        'loss': avg_loss,
        **full_metrics
    }
