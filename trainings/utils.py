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
        checkpoint_path: Path to save best model
        early_stopping_patience: Number of epochs to wait before early stopping
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(checkpoint_dir, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    scaler = GradScaler()

    history = {
        'train_loss': [],
        'valid_loss': [],
        'epoch_times': []
    }
    best_valid_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        epoch_start_time = datetime.now()
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_data_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attn_mask'].to(device)
            span_indices = batch['span_indices']
            synset_ids = batch['synset_ids'].to(device)

            optimizer.zero_grad()

            with autocast(device_type=device):
                outputs = model(input_ids, attention_mask, span_indices=span_indices)
                loss = loss_fn(outputs, synset_ids)  
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Calculate metrics
            running_loss += loss.item()


        # Calculate training metrics
        epoch_time = (datetime.now() - epoch_start_time).total_seconds()
        train_loss = running_loss / len(train_data_loader)


        valid_loss = evaluate_model(model, valid_data_loader, loss_fn, device)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        history['epoch_times'].append(epoch_time)

        if epoch % ckpt_interval == 0:
            checkpoint_path = os.path.join(run_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, checkpoint_path)

        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        
        # Checkpoint saving and early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_path = os.path.join(run_dir, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
            print(f"  New best model saved to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping after {epoch+1} epochs!")
                break
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"  Time: {epoch_time:.2f}s")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Valid Loss: {valid_loss:.4f}")
        print(f"  Early stopping counter: {patience_counter}/{early_stopping_patience}")

    final_model_path = os.path.join(run_dir, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    
    history_path = os.path.join(run_dir, "training_history.pt")
    torch.save(history, history_path)
    
    print(f"\nTraining completed. Best validation accuracy: {best_valid_loss:.4f}")
    print(f"All checkpoints and models saved to: {run_dir}")
    
    return history, model


def evaluate_model(model, data_loader, loss_fn, device):
    """Enhanced evaluation with detailed metrics"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attn_mask'].to(device)
            span_indices = batch['span_indices']  
            synset_ids = batch['synset_ids'].to(device)
            
            outputs = model(input_ids, attention_mask, span_indices=span_indices)
            loss = loss_fn(outputs, synset_ids)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += synset_ids.size(0)
            correct += (predicted == synset_ids).sum().item()
    
    avg_loss = running_loss / len(data_loader)
    
    return avg_loss