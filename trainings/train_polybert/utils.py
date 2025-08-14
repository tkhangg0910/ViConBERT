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
    
    print(f"[train] run_dir: {run_dir}  | device: {device} | AMP: {use_amp}")

    for epoch in range(num_epochs):
        epoch_start = datetime.now()
        model.train()
        train_tp = train_fp = train_fn = 0

        running_loss = 0.0
        train_steps = 0

        pbar = tqdm(train_data_loader, 
                         desc=f"Training Epoch {epoch+1}/{num_epochs}",
                         position=0, leave=True)
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
            tp_batch = (pred_sense_ids == correct_labels_in_batch).sum().item()
            # FP: dự đoán sai
            fp_batch = (pred_sense_ids != correct_labels_in_batch).sum().item()
            # FN: trong WSD mỗi sample có đúng 1 label, nên FN = FP trong trường hợp single-label
            fn_batch = fp_batch

            train_tp += tp_batch
            train_fp += fp_batch
            train_fn += fn_batch


            # logging
            if global_step % metric_log_interval == 0:
                postfix = {"Loss": f"{(running_loss / max(1, train_steps)):.4f}"}
                if grad_clip:
                    postfix["GradMax"] = f"{grad_clipper.max_norm:.2f}"
                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"loss": f"{loss.item()*grad_accum_steps:.4f}"})
        precision = train_tp / (train_tp + train_fp + 1e-8)
        recall    = train_tp / (train_tp + train_fn + 1e-8)
        f1        = 2 * precision * recall / (precision + recall + 1e-8)

        train_metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

        
        avg_train_loss = running_loss / max(1, train_steps)
        history["train_loss"].append(avg_train_loss)

        # ============= VALIDATION =============
        model.eval()
        TP, FP, FN = 0, 0, 0
        valid_loss = 0.0
        valid_steps = 0
        loss_fn_val = torch.nn.CrossEntropyLoss()

        with torch.no_grad():
            eval_pbar = tqdm(valid_data_loader, desc="Evaluating", position=1, leave=True)
            for batch in eval_pbar:
                # Context
                context_input_ids = batch["context_input_ids"].to(device)
                context_attn_mask = batch["context_attn_mask"].to(device)
                target_spans = batch["target_spans"].to(device)
                gold_glosses = batch["gold_glosses"]
                cand_counts = batch["candidate_gloss_counts"]

                # Encode context
                rwt = model.forward_context(context_input_ids, context_attn_mask, target_spans)
                rwt_mean = rwt.mean(dim=1)

                # Encode gloss candidates
                flat_inp = batch["candidate_gloss_flat_input_ids"]
                flat_att = batch["candidate_gloss_flat_attn"]
                if flat_inp is None:
                    continue

                flat_inp = flat_inp.to(device)
                flat_att = flat_att.to(device)
                rFg_flat = model.forward_gloss(flat_inp, flat_att)
                rFg_flat_mean = rFg_flat.mean(dim=1)

                # Compare predictions
                start = 0
                batch_size = rwt_mean.size(0)
                for i in range(batch_size):
                    cnt = cand_counts[i]
                    end = start + cnt
                    cand_embs = rFg_flat_mean[start:end]
                    scores = torch.matmul(rwt_mean[i:i+1], cand_embs.T).squeeze(0).cpu()
                    pred_idx = int(torch.argmax(scores).item())
                    gold_idx = batch["candidate_glosses_grouped"][i].index(gold_glosses[i])

                    pred_gloss = batch["candidate_glosses_grouped"][i][pred_idx]
                    gold_gloss = gold_glosses[i]

                    if pred_gloss == gold_gloss:
                        TP += 1
                    else:
                        FP += 1
                        FN += 1  
                    loss = loss_fn_val(scores.unsqueeze(0),, torch.tensor([gold_idx], device=scores.device))
                    valid_loss += loss.item()

                    valid_steps += 1
                    start = end

        # Precision, Recall, F1
        valid_precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        valid_recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        valid_f1 = 2 * valid_precision * valid_recall / (valid_precision + valid_recall) if (valid_precision + valid_recall) > 0 else 0.0



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
