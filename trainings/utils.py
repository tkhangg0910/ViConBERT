import os
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from datetime import datetime
from utils.metrics import compute_step_metrics, compute_full_metrics_large_scale, ndcg_step_metrics, ndcg_full_metrics

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
        'train_soft_metrics': [],
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
        train_soft_accum   = {f'soft_recall@{k}': 0.0 for k in metric_k_vals}
        train_soft_accum.update({f'soft_precision@{k}': 0.0 for k in metric_k_vals})

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
                if model.encoder_type == "attentive":
                    gloss_embd = gloss_embd.repeat(outputs.size(0), 1, 1)
                    P, B, D = gloss_embd.size()
                    
                    gloss_embd = gloss_embd.permute(1,0,2).reshape(B, P * D)
                    
                    outputs = outputs.permute(1,0,2).reshape(B, P * D)
                loss = loss_fn(outputs,gloss_embd, synset_ids)
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Calculate training metrics
            running_loss += loss.item()
            

            batch_metrics = compute_step_metrics(outputs, synset_ids, 
                                               k_vals=metric_k_vals, 
                                               device=device)
            soft_metrics  = ndcg_step_metrics(outputs,gloss_embd, synset_ids, metric_k_vals, device=device)

            for k in metric_k_vals:
                train_metrics_accum[f'recall@{k}']    += batch_metrics[f'recall@{k}']
                train_metrics_accum[f'precision@{k}'] += batch_metrics[f'precision@{k}']
                train_soft_accum[f'ndcg@{k}']    += soft_metrics[f'ndcg@{k}']


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
            
            # if scheduler:
            #     scheduler.step() 
                
        train_metrics = {}
        num_batches = len(train_data_loader)
        train_metrics      = {f'recall@{k}': train_metrics_accum[f'recall@{k}'] / num_batches
                              for k in metric_k_vals}
        train_metrics.update({f'precision@{k}': train_metrics_accum[f'precision@{k}'] / num_batches
                              for k in metric_k_vals})
        train_soft_metrics = {f'ndcg@{k}': train_soft_accum[f'ndcg@{k}'] / num_batches
                              for k in metric_k_vals}
        train_soft_metrics.update({f'ndcg@{k}': train_soft_accum[f'ndcg@{k}'] / num_batches
                              for k in metric_k_vals})

        train_loss = running_loss / num_batches
        train_metrics['loss'] = train_loss
        history['train_metrics'].append(train_metrics)
        
        history['train_metrics'].append(train_metrics)
        history['train_soft_metrics'].append(train_soft_metrics)

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
                f"SoftR@{k}:      {train_soft_metrics[f'soft_recall@{k}']:.4f} | "
                f"SoftP@{k}:      {train_soft_metrics[f'soft_precision@{k}']:.4f}"
            )

        # Validation metrics
        print("\n  VALIDATION METRICS:")
        print(f"    Loss: {valid_metrics['loss']:.4f}")
        for k in metric_k_vals:
            print(
                f"    Recall@{k}:     {valid_metrics[f'recall@{k}']:.4f} | "
                f"Precision@{k}:  {valid_metrics[f'precision@{k}']:.4f} | "
                f"F1@{k}:         {valid_metrics[f'f1@{k}']:.4f} | "
                f"SoftR@{k}:      {valid_metrics[f'soft_recall@{k}']:.4f} | "
                f"SoftP@{k}:      {valid_metrics[f'soft_precision@{k}']:.4f}"
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


def evaluate_model(model, data_loader, loss_fn, device, metric_k_vals=(1,5,10)):
    model.eval()
    running_loss = 0.0
    all_embs = []
    all_labels = []

    # 1) Lấy full gloss embeddings và build FAISS index
    ds = data_loader.dataset
    # Giả sử ds.global_label_to_synset (label→synset_id) và ds.gloss_embeddings (synset_id→tensor)
    S = len(ds.global_label_to_synset)
    D = next(iter(ds.gloss_embeddings.values())).size(0)
    # Tạo matrix [S, D] theo thứ tự label 0..S-1
    G = torch.stack([
        ds.gloss_embeddings[ ds.global_label_to_synset[i] ]
        for i in range(S)
    ]).to(device)  # [S, D]
    # normalize và đưa qua FAISS
    import faiss
    G_np = torch.nn.functional.normalize(G, dim=1).cpu().numpy().astype('float32')
    index = faiss.IndexFlatIP(D)
    index.add(G_np)

    with torch.inference_mode():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False, ascii=True):
            gloss_embd = batch["gloss_embd"].to(device)
            cid = batch["context_input_ids"].to(device)
            cam = batch["context_attn_mask"].to(device)
            spans = batch.get("target_spans", None)
            if spans is not None:
                spans = spans.to(device)
            labels = batch["synset_ids"].to(device)

            with autocast(device_type=device):
                out = model({"input_ids": cid, "attention_mask": cam}, target_span=spans)
                # Nếu attentive encoder: flatten polym dim
                if model.encoder_type == "attentive":
                    P, B, d = gloss_embd.size()
                    gloss_embd = gloss_embd.permute(1,0,2).reshape(B, P*d)
                    out = out.permute(1,0,2).reshape(B, P*d)

                loss = loss_fn(out, gloss_embd, labels)

            running_loss += loss.item()
            all_embs.append(out.cpu())
            all_labels.append(labels.cpu())

    # 2) Concatenate
    C_all = torch.cat(all_embs, dim=0)     # [N, D]
    L_all = torch.cat(all_labels, dim=0)   # [N]
    avg_loss = running_loss / len(data_loader)

    # 3) Hard metrics (recall/precision/F1) – nếu vẫn cần
    hard = compute_full_metrics_large_scale(C_all, L_all, k_vals=metric_k_vals, device=device)

    # 4) nDCG metrics
    # Tính chunk_size tự động như trước
    N = C_all.size(0)
    if device.startswith('cuda'):
        total_mem = torch.cuda.get_device_properties(device).total_memory
        free_mem  = total_mem - torch.cuda.memory_allocated(device)
        mem_per_row = 4 * D * N
        chunk_size = max(1, min(int(free_mem / mem_per_row), 5000))
    else:
        chunk_size = 1000

    ndcg = ndcg_full_metrics(
        context_emb=C_all,
        labels=L_all,
        k_vals=metric_k_vals,
        faiss_index=index,
        batch_size=chunk_size,
        device=device
    )

    return {
        'loss': avg_loss,
        **hard,
        **ndcg
    }
