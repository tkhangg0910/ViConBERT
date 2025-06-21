import os
import torch
from tqdm import tqdm
from torch.amp import GradScaler, autocast
from datetime import datetime

def train_model(num_epochs, train_data_loader, valid_data_loader, 
                loss_fn, optimizer, model, device, 
                checkpoint_dir,
                scheduler=None,
                early_stopping_patience=3,
                ckpt_interval=10
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
        'valid_loss': [],
        'valid_accuracy': [],
        'epoch_times': []
    }
    best_valid_loss = float('inf')
    patience_counter = 0
    
    print(f"Training started. Checkpoints will be saved to: {run_dir}")
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        
        # ======================
        # TRAINING PHASE
        # ======================
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training loop
        train_pbar = tqdm(train_data_loader, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}",
                         position=0, leave=True)
        
        for batch_idx, batch in enumerate(train_pbar):
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

            # Calculate training metrics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += synset_ids.size(0)
            train_correct += (predicted == synset_ids).sum().item()
            
            # Update progress bar
            current_loss = running_loss / (batch_idx + 1)
            current_acc = 100. * train_correct / train_total if train_total > 0 else 0
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        # ======================
        # VALIDATION PHASE
        # ======================
        print(f"\nValidating epoch {epoch+1}...")
        valid_loss, valid_accuracy = evaluate_model(model, valid_data_loader, loss_fn, device)
        
        # Calculate final training metrics
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        train_loss = running_loss / len(train_data_loader)
        train_accuracy = 100. * train_correct / train_total if train_total > 0 else 0

        # Update history
        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['valid_accuracy'].append(valid_accuracy)
        history['epoch_times'].append(epoch_time)

        # Learning rate scheduling
        if scheduler:
            scheduler.step(valid_loss)  # For ReduceLROnPlateau
            # scheduler.step()  # For other schedulers

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
                'train_loss': train_loss,
                'train_accuracy': train_accuracy,
                'valid_loss': valid_loss,
                'valid_accuracy': valid_accuracy,
                'history': history
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # ======================
        # EARLY STOPPING & BEST MODEL SAVING
        # ======================
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_path = os.path.join(run_dir, "best_model.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_valid_loss': best_valid_loss,
                'valid_accuracy': valid_accuracy,
                'history': history
            }, best_model_path)
            patience_counter = 0
            print(f"✓ New best model saved! Valid Loss: {valid_loss:.4f}")
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
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}%")
        print(f"  Valid Loss: {valid_loss:.4f} | Valid Acc: {valid_accuracy:.2f}%")
        print(f"  Early stopping: {patience_counter}/{early_stopping_patience}")
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
        'final_train_loss': train_loss,
        'final_valid_loss': valid_loss,
        'best_valid_loss': best_valid_loss,
        'history': history
    }, final_model_path)
    
    history_path = os.path.join(run_dir, "training_history.pt")
    torch.save(history, history_path)
    
    print(f"\n🎉 Training completed!")
    print(f"📊 Best validation loss: {best_valid_loss:.4f}")
    print(f"💾 All files saved to: {run_dir}")
    
    return history, model


def evaluate_model(model, data_loader, loss_fn, device):
    """Enhanced evaluation with detailed metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        eval_pbar = tqdm(data_loader, desc="Evaluating", position=0, leave=False)
        for batch in eval_pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attn_mask'].to(device)
            span_indices = batch['span_indices'].to(device)  # Fixed: move to device
            synset_ids = batch['synset_ids'].to(device)
            
            with autocast(device_type=device):
                outputs = model(input_ids, attention_mask, span_indices=span_indices)
                loss = loss_fn(outputs, synset_ids)
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += synset_ids.size(0)
            correct += (predicted == synset_ids).sum().item()
            
            # Update progress bar
            current_acc = 100. * correct / total if total > 0 else 0
            eval_pbar.set_postfix({'Acc': f'{current_acc:.2f}%'})
    
    avg_loss = running_loss / len(data_loader)
    accuracy = 100. * correct / total if total > 0 else 0
    
    return avg_loss, accuracy
