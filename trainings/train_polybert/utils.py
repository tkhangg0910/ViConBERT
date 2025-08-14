import os
import torch
from tqdm import tqdm
import numpy as np
from torch.amp import GradScaler, autocast
from datetime import datetime
from utils.metrics import compute_precision_recall_f1_for_wsd
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
        "epoch_times": []
    }

    best_valid_loss = float("inf")
    patience_counter = 0
    global_step = 0

    model.to(device)
    train_metrics_accum={
        "recall":0.0,
        "precision":0.0,
        "f1":0.0
    }
    print(f"[train] run_dir: {run_dir}  | device: {device} | AMP: {use_amp}")

    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        model.train()

        running_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_data_loader, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}",
                         position=0, leave=False)
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            global_step += 1
            train_steps += 1

            context_inputs = {
                "input_ids": batch["context_input_ids"].to(device),
                "attention_mask": batch["context_attn_mask"].to(device)
            }
            gloss_inputs = {
                "input_ids": batch["gloss_input_ids"].to(device),
                "attention_mask": batch["gloss_attn_mask"].to(device)
            }
            target_idx = batch["target_spans"].to(device)  # shape [B]

            synset_ids = batch.get("synset_ids", None)
            if synset_ids is not None:
                synset_ids = synset_ids.to(device)

            # forward + loss (use AMP if available)
            with autocast(device_type=device):
                model_loss, MF = model(context_inputs, gloss_inputs, target_idx)
                # allow user to override loss (e.g., add reg or custom objective)
                if loss_fn is not None:
                    loss = loss_fn((model_loss, MF), batch, device)
                else:
                    loss = model_loss

                loss = loss / float(grad_accum_steps)

            scaler.scale(loss).backward()

            # gradient accumulation step
            if (batch_idx + 1) % grad_accum_steps == 0:
                # unscale for clipping
                scaler.unscale_(optimizer)
                if grad_clip:
                    current_norm = grad_clipper.clip(model)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    # scheduler.step() policy depends on scheduler (per-step or per-epoch)
                    try:
                        scheduler.step()
                    except Exception:
                        pass

            running_loss += loss.item() * float(grad_accum_steps)  # accumulate true loss
            # optional metrics
            batch_size = MF.size(0)
            correct_labels_in_batch = torch.arange(batch_size, device=MF.device)
            pred_sense_ids = MF.argmax(dim=1)
            precision, recall, f1 = compute_precision_recall_f1_for_wsd(pred_sense_ids, correct_labels_in_batch)
            train_metrics_accum["f1"]+=f1
            train_metrics_accum["precision"]+=precision
            train_metrics_accum["recall"]+=recall

            # logging
            if global_step % metric_log_interval == 0:
                postfix = {"Loss": f"{(running_loss / max(1, train_steps)):.4f}"}
                if grad_clip:
                    postfix["GradMax"] = f"{grad_clipper.max_norm:.2f}"
                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"loss": f"{loss.item()*grad_accum_steps:.4f}"})
        pbar.close()
        print("Finish")
        num_batches = len(train_data_loader)
        train_metrics = {}
        train_metrics = {'recall': train_metrics_accum['recall'] / num_batches}
        train_metrics.update({'precision': train_metrics_accum['precision'] / num_batches})
        train_metrics.update({'f1': train_metrics_accum['f1'] / num_batches})
        
        avg_train_loss = running_loss / max(1, train_steps)
        history["train_loss"].append(avg_train_loss)

        # ============= VALIDATION =============
        model.eval()
        valid_precision_accum, valid_recall_accum, valid_f1_accum = 0.0, 0.0, 0.0
        valid_steps = 0
        valid_loss = 0.0
        with torch.no_grad():
            eval_pbar = tqdm(valid_data_loader, desc="Evaluating", position=1, leave=True)
            for batch in eval_pbar:
                batch_size = len(batch["context_input_ids"])  
                for i in range(batch_size):
                    context_inputs = {
                        "input_ids": batch["context_input_ids"][i].unsqueeze(0).to(device),
                        "attention_mask": batch["context_attn_mask"][i].unsqueeze(0).to(device)
                    }
                    target_idx = batch["target_spans"][i].unsqueeze(0).to(device)
                    candidate_glosses = batch["candidate_glosses"][i]
                    gloss_tok = model.tokenizer(
                        candidate_glosses,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=256
                    )
                    gloss_inputs = {
                        "input_ids": gloss_tok["input_ids"].to(device),
                        "attention_mask": gloss_tok["attention_mask"].to(device)
                    }


                    with autocast(device_type=device):
                        _, MF = model(context_inputs, gloss_inputs, target_idx)  #

                    pred_idx = MF.squeeze(0).argmax().item()
                    pred_gloss = candidate_glosses[pred_idx]

                    gold_gloss = batch["gloss"][i]

                    p, r, f = compute_precision_recall_f1_for_wsd(
                        torch.tensor([pred_gloss]),
                        torch.tensor([gold_gloss])
                    )
                    valid_precision_accum += p
                    valid_recall_accum += r
                    valid_f1_accum += f
                    valid_steps += 1

        # Average metrics
        valid_precision = valid_precision_accum / valid_steps
        valid_recall = valid_recall_accum / valid_steps
        valid_f1 = valid_f1_accum / valid_steps

        avg_valid_loss = valid_loss / max(1, valid_steps)
        history["valid_loss"].append(avg_valid_loss)

        epoch_time = (datetime.now() - epoch_start).total_seconds()
        history["epoch_times"].append(epoch_time)
        print("=======================================TRAIN=======================================")        
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | time: {epoch_time:.1f}s")
        print(f"Train Recall: {train_metrics['recall']:.4f} | Train Precision: {train_metrics['precision']:.4f} | Train F1: {train_metrics['f1']:.4f} | ")
        print("=======================================VALID========================================")
        print(f"[Epoch {epoch+1}] Valid Loss: {avg_valid_loss:.4f} | time: {epoch_time:.1f}s")
        print(f"Train Recall: {valid_recall:.4f} | Train Precision: {valid_precision:.4f} | Train F1: {valid_f1:.4f} | ")

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
