import os
import torch
from tqdm import tqdm
import numpy as np
from torch.amp import GradScaler, autocast
from datetime import datetime
import torch.nn.functional as F
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

            context_inputs = {
                "input_ids": batch["context_input_ids"].to(device),
                "attention_mask": batch["context_attn_mask"].to(device)
            }
            gloss_inputs = {
                "input_ids": batch["gloss_input_ids"].to(device),
                "attention_mask": batch["gloss_attn_mask"].to(device)
            }
            target_idx = batch["target_spans"].to(device)  # shape [B]

            word_id = batch.get("word_id", None)
            if word_id is not None:
                word_id = word_id.to(device)

            # forward + loss (use AMP if available)
            with autocast(device_type=device):
                rF_wt, rF_g = model(context_inputs, gloss_inputs, target_idx)
                # allow user to override loss (e.g., add reg or custom objective)
                loss, MF = model.batch_contrastive_loss(rF_wt, rF_g, word_id)

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
            B = MF.size(0)
            pred_indices = MF.argmax(dim=1)  # [B] predicted most similar gloss in batch

            # batch-level ground truth: samples with same word_id are positives
            pos_mask = (word_id.unsqueeze(0) == word_id.unsqueeze(1)).cpu().numpy()  # [B,B]
            batch_tp, batch_fp, batch_fn = 0, 0, 0
            for i in range(B):
                # predicted positive = pred_indices[i]
                if pos_mask[i, pred_indices[i]]:
                    batch_tp += 1
                else:
                    batch_fp += 1
                batch_fn += pos_mask[i].sum() - (1 if pos_mask[i, pred_indices[i]] else 0)

            train_tp += batch_tp
            train_fp += batch_fp
            train_fn += batch_fn

            batch_precision = batch_tp / (batch_tp + batch_fp + 1e-8)
            batch_recall = batch_tp / (batch_tp + batch_fn + 1e-8)
            batch_f1 = 2 * batch_precision * batch_recall / (batch_precision + batch_recall + 1e-8)

            # logging
            if global_step % metric_log_interval == 0:
                postfix = {"Loss": f"{(running_loss / max(1, train_steps)):.4f}"}
                if grad_clip:
                    postfix["GradMax"] = f"{grad_clipper.max_norm:.2f}"
                pbar.set_postfix(postfix)
            else:
                pbar.set_postfix({"loss": f"{loss.item()*grad_accum_steps:.4f}"})
                
        precision = batch_precision/len(train_data_loader)
        recall = batch_recall/len(train_data_loader)
        f1 = 2 * batch_f1/len(train_data_loader)

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
                batch_size = len(batch["context_input_ids"])
                context_inputs = {
                    "input_ids": batch["context_input_ids"].to(device),
                    "attention_mask": batch["context_attn_mask"].to(device)
                }

                target_spans = batch["target_spans"].to(device)
                rF_wt = model.forward_context(context_inputs["input_ids"],
                                            context_inputs["attention_mask"],
                                            target_spans)  # [B, polym, H]

                for i in range(batch_size):
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
        print("=======================================TRAIN=======================================")        
        print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | time: {epoch_time:.1f}s")
        print(f"Train Recall: {train_metrics['recall']:.4f} | Train Precision: {train_metrics['precision']:.4f} | Train F1: {train_metrics['f1']:.4f} | ")
        print("=======================================VALID========================================")
        print(f"[Epoch {epoch+1}] Valid Loss: {avg_valid_loss:.4f} | time: {epoch_time:.1f}s")
        print(f"Valid Recall: {valid_recall:.4f} | Valid Precision: {valid_precision:.4f} | Valid F1: {valid_f1:.4f} | ")

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
import random
import math
from collections import defaultdict, deque
from torch.utils.data import Sampler

class GlossAwareBatchSampler(Sampler):
    """
    Batch sampler that:
      - tries to avoid multiple samples from the same gloss in one batch
      - encourages (with probability encourage_target_prob) to include multiple
        samples from the same target_word but with different glosses
      - still yields every index exactly once per epoch (will allow same-gloss
        duplicates only when necessary)
    Args:
      dataset: dataset instance, must expose `all_samples` list with dict items
               containing keys "target_word" and "gloss".
      batch_size: int
      shuffle: whether to shuffle order each epoch
      encourage_target_prob: float in [0,1], probability to try target-oriented fill
      seed: optional random seed for reproducibility
    """
    def __init__(self, dataset, batch_size, shuffle=True, encourage_target_prob=0.6, seed=None):
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.encourage_target_prob = float(encourage_target_prob)
        self.seed = seed

        # build mappings target_word -> gloss -> list(indices)
        self.target2gloss2idxs = defaultdict(lambda: defaultdict(list))
        self.gloss2idxs = defaultdict(list)

        for idx, s in enumerate(self.dataset.all_samples):
            target = s["target_word"]
            gloss = s["gloss"]
            self.target2gloss2idxs[target][gloss].append(idx)
            self.gloss2idxs[gloss].append(idx)

        self.num_samples = len(self.dataset)

    def __len__(self):
        return math.ceil(self.num_samples / self.batch_size)

    def _make_epoch_queues(self, rnd):
        """Return mutable deques for gloss queues and target->gloss->deque mapping."""
        gloss_queues = {g: deque(idxs) for g, idxs in self.gloss2idxs.items()}
        # optionally shuffle order within each gloss
        if self.shuffle:
            for g in gloss_queues:
                idxs = list(gloss_queues[g])
                rnd.shuffle(idxs)
                gloss_queues[g] = deque(idxs)

        target_queues = {}
        for t, gmap in self.target2gloss2idxs.items():
            inner = {}
            for g, idxs in gmap.items():
                idxs_copy = list(idxs)
                if self.shuffle:
                    rnd.shuffle(idxs_copy)
                inner[g] = deque(idxs_copy)
            target_queues[t] = inner

        return gloss_queues, target_queues

    def __iter__(self):
        # use local Random for reproducibility per-epoch
        rnd = random.Random(self.seed)
        if self.shuffle:
            # change seed each epoch slightly if provided
            rnd.seed(None)

        gloss_queues, target_queues = self._make_epoch_queues(rnd)

        # helper to know whether any items remain
        def any_remaining():
            for q in gloss_queues.values():
                if q:
                    return True
            return False

        # list of gloss keys (static) for sampling order
        gloss_keys = list(gloss_queues.keys())

        while any_remaining():
            batch = []
            used_glosses = set()

            # 1) Optionally try to pick multiple different-gloss samples from same target
            if rnd.random() < self.encourage_target_prob:
                # collect candidate targets which have >=2 distinct gloss queues non-empty
                candidates = []
                for t, gmap in target_queues.items():
                    nonempty_glosses = [g for g, dq in gmap.items() if dq]
                    if len(nonempty_glosses) >= 2:
                        candidates.append(t)
                if candidates:
                    t = rnd.choice(candidates)
                    available_glosses = [g for g, dq in target_queues[t].items() if dq]
                    rnd.shuffle(available_glosses)
                    for g in available_glosses:
                        if len(batch) >= self.batch_size:
                            break
                        idx = target_queues[t][g].popleft()
                        batch.append(idx)
                        used_glosses.add(g)
                        # also pop the same idx from gloss_queues (if present)
                        # (gloss_queues should contain the same idx list; remove by value)
                        try:
                            gloss_queues[g].remove(idx)
                        except ValueError:
                            # might have been removed already (rare); ignore
                            pass

            # 2) Fill remaining slots by sampling from glosses not yet used in this batch
            remaining_slots = self.batch_size - len(batch)
            if remaining_slots > 0:
                # create list of glosses that still have items and are not used in this batch
                candidates = [g for g, dq in gloss_queues.items() if dq and g not in used_glosses]
                # shuffle candidate gloss list to diversify
                rnd.shuffle(candidates)
                for g in candidates:
                    if len(batch) >= self.batch_size:
                        break
                    idx = gloss_queues[g].popleft()
                    batch.append(idx)
                    used_glosses.add(g)
                    # also remove from target_queues where it belongs
                    # (we can avoid search by trusting later steps, but try to remove to keep consistency)
                    # attempt removal from target_queues:
                    # (each idx is in exactly one target->gloss queue)
                    # loop small maps: cost acceptable
                    for t, gmap in target_queues.items():
                        if g in gmap:
                            if idx in gmap[g]:
                                try:
                                    gmap[g].remove(idx)
                                except ValueError:
                                    pass
                            break

            # 3) If still not full, allow picking from glosses already used (i.e., duplicates of gloss)
            if len(batch) < self.batch_size:
                candidates = [g for g, dq in gloss_queues.items() if dq]
                rnd.shuffle(candidates)
                for g in candidates:
                    if len(batch) >= self.batch_size:
                        break
                    idx = gloss_queues[g].popleft()
                    batch.append(idx)
                    # try remove from target_queues too
                    for t, gmap in target_queues.items():
                        if g in gmap:
                            if idx in gmap[g]:
                                try:
                                    gmap[g].remove(idx)
                                except ValueError:
                                    pass
                            break

            # final sanity: if batch empty break (shouldn't happen)
            if not batch:
                break

            yield batch
