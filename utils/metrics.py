import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch.nn.functional as F

def normalize_embeddings(embeddings):
    """Normalize embeddings for efficient cosine similarity computation"""
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def compute_cosine_similarity(embeddings):
    """Compute the cosine similarity matrix efficiently"""
    embeddings = normalize_embeddings(embeddings)
    return torch.mm(embeddings, embeddings.t())

def recall_at_k_batch(embeddings, labels, k=5, device='cuda'):
    """
    Efficiently compute Recall@K for a batch of data
    embeddings: Tensor [batch_size, D]
    labels: Tensor [batch_size] containing synset IDs
    """
    batch_size = embeddings.size(0)
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # Compute similarity matrix within the batch
    sim = torch.mm(normalize_embeddings(embeddings), 
                   normalize_embeddings(embeddings).t())
    
    # Mask self-similarity
    mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    sim.masked_fill_(mask, -10.0)
    
    # Get top-k most similar items
    _, topk_indices = torch.topk(sim, k, dim=1)
    
    recall_count = 0
    for i in range(batch_size):
        if labels[i] in labels[topk_indices[i]]:
            recall_count += 1
            
    return recall_count / batch_size

def precision_at_k_batch(embeddings, labels, k=5, device='cuda'):
    """
    Efficiently compute Precision@K for a batch of data
    embeddings: Tensor [batch_size, D]
    labels: Tensor [batch_size] containing synset IDs
    """
    batch_size = embeddings.size(0)
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # Compute similarity matrix within the batch
    sim = torch.mm(normalize_embeddings(embeddings), 
                   normalize_embeddings(embeddings).t())
    
    # Mask self-similarity
    mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    sim.masked_fill_(mask, -10.0)
    
    # Get top-k most similar items
    _, topk_indices = torch.topk(sim, k, dim=1)
    
    precision_sum = 0.0
    for i in range(batch_size):
        correct = (labels[topk_indices[i]] == labels[i]).sum().item()
        precision_sum += correct / k
        
    return precision_sum / batch_size

def compute_step_metrics(embeddings, labels, k_vals=(1, 5, 10), device='cuda'):
    """
    Quickly compute evaluation metrics for a training step
    Returns a dictionary of computed metrics
    """
    metrics = {}
    
    for k in k_vals:
        metrics[f'recall@{k}'] = recall_at_k_batch(embeddings, labels, k, device=device)
        metrics[f'precision@{k}'] = precision_at_k_batch(embeddings, labels, k, device=device)
    
    return metrics


def soft_step_metrics(embeddings: torch.Tensor,
                      labels: torch.Tensor,
                      k_vals=(1,5,10),
                      theta: float = 0.5,
                      device='cuda'):
    """
    embeddings: [B, D], labels: [B]
    Trả về dict với các keys: soft_recall@k, soft_precision@k
    """
    B, D = embeddings.shape
    emb = embeddings.to(device)
    lbl = labels.to(device)
    emb = F.normalize(emb, p=2, dim=1)

    # compute sim matrix within batch
    sim = torch.mm(emb, emb.t())                   # [B,B]
    diag = torch.eye(B, dtype=torch.bool, device=device)
    sim.masked_fill_(diag,  -float('inf'))

    metrics = {}
    for k in k_vals:
        # top-k indices
        _, topk = sim.topk(k, dim=1)               # [B,k]

        soft_rec, soft_prec = 0.0, 0.0
        for i in range(B):
            neigh = topk[i]                        # k neighbor idx
            # get cosine sims to each neighbor
            sims = sim[i, neigh]
            # soft hits mask
            hits = sims >= theta                   # bool[k]
            # recall: có ít nhất 1 hit?
            soft_rec += hits.any().float().item()
            # precision: tỉ lệ hits
            soft_prec += hits.float().sum().item() / k

        metrics[f'soft_recall@{k}']    = soft_rec / B
        metrics[f'soft_precision@{k}'] = soft_prec / B

    return metrics


def recall_at_k_full(embeddings, labels, k=5, batch_size=1000, device='cuda'):
    """Compute Recall@K over the full dataset with batch processing"""
    n = len(labels)
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    recall_count = 0
    num_batches = (n + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        batch_size_actual = end_idx - start_idx
        
        # Compute similarity between current batch and full dataset
        batch_emb = embeddings[start_idx:end_idx]
        sim_batch = torch.mm(normalize_embeddings(batch_emb), 
                             normalize_embeddings(embeddings).t())
        
        # Mask self-similarity
        for j in range(batch_size_actual):
            idx = start_idx + j
            sim_batch[j, idx] = -10.0
            
        # Get top-k for each sample in the batch
        _, topk_indices = torch.topk(sim_batch, k, dim=1)
        
        # Check recall for each sample
        for j in range(batch_size_actual):
            idx = start_idx + j
            if labels[idx] in labels[topk_indices[j]]:
                recall_count += 1
                
    return recall_count / n

def precision_at_k_full(embeddings, labels, k=5, batch_size=1000, device='cuda'):
    """Compute Precision@K over the full dataset with batch processing"""
    n = len(labels)
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    precision_sum = 0.0
    num_batches = (n + batch_size - 1) // batch_size
    
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, n)
        batch_size_actual = end_idx - start_idx
        
        # Compute similarity between current batch and full dataset
        batch_emb = embeddings[start_idx:end_idx]
        sim_batch = torch.mm(normalize_embeddings(batch_emb), 
                             normalize_embeddings(embeddings).t())
        
        # Mask self-similarity
        for j in range(batch_size_actual):
            idx = start_idx + j
            sim_batch[j, idx] = -10.0
            
        # Get top-k for each sample in the batch
        _, topk_indices = torch.topk(sim_batch, k, dim=1)
        
        # Compute precision for each sample
        for j in range(batch_size_actual):
            idx = start_idx + j
            correct = (labels[topk_indices[j]] == labels[idx]).sum().item()
            precision_sum += correct / k
                
    return precision_sum / n

def normalized_mutual_info(embeddings, labels, max_samples=10000, random_state=0):
    """Compute Normalized Mutual Information (NMI)"""
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Subsample if dataset is too large
    if len(labels) > max_samples:
        indices = np.random.choice(len(labels), max_samples, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    # Number of clusters = number of unique synsets
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    # Avoid case where there is only one cluster
    if n_clusters <= 1:
        return 0.0
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return normalized_mutual_info_score(labels, cluster_labels)

def compute_full_metrics_large_scale(embeddings, labels, k_vals, device='cuda'):
    embeddings = normalize_embeddings(embeddings)
    n = len(embeddings)
    
    unique_labels, label_counts = torch.unique(labels, return_counts=True)
    label_count_dict = dict(zip(unique_labels.tolist(), label_counts.tolist()))
    
    # Auto chunk size based on GPU memory
    if device == 'cuda':
        total_mem = torch.cuda.get_device_properties(device).total_memory
        used_mem = torch.cuda.memory_allocated(device)
        free_mem = total_mem - used_mem
        mem_per_row = 4 * embeddings.shape[1] * n
        chunk_size = max(1, int(free_mem / mem_per_row))
        chunk_size = min(chunk_size, 5000)  
    else:
        chunk_size = 1000
    
    max_k = max(k_vals)
    topk_indices = torch.empty((n, max_k), dtype=torch.long, device='cpu')
    
    for i in tqdm(range(0, n, chunk_size), desc="Computing similarity",ascii=True):
        start_i = i
        end_i = min(i + chunk_size, n)
        chunk = embeddings[start_i:end_i].to(device)
        full_emb = embeddings.to(device)
        
        sim = torch.mm(chunk, full_emb.t())
        
        # Vectorized diagonal masking
        rows = torch.arange(chunk.size(0), device=device)
        cols = torch.arange(start_i, end_i, device=device)
        sim[rows, cols] = -float('inf')
        
        _, chunk_topk_indices = torch.topk(sim, max_k, dim=1)
        topk_indices[start_i:end_i] = chunk_topk_indices.cpu()
    
    labels_cpu = labels.cpu()
    
    metrics = {}
    for k in k_vals:
        recall_sum = 0.0
        precision_sum = 0.0
        valid_samples = 0 
        
        for i in tqdm(range(n), desc=f"Computing metrics@k={k}",ascii=True):
            current_label = labels_cpu[i].item()
            current_count = label_count_dict.get(current_label, 0)
            topk = topk_indices[i, :k]
            matches = (labels_cpu[topk] == labels_cpu[i])
            
            recall_sum += matches.any().item()
            
            if current_count > 1: 
                actual_k = min(k, current_count - 1)  
                precision_sum += matches.float().sum().item() / actual_k
                valid_samples += 1
        recall = recall_sum / n
        metrics[f'recall@{k}'] = recall
        
        if valid_samples > 0:
            precision = precision_sum / valid_samples
        else:
            precision = 0.0
        metrics[f'precision@{k}'] = precision
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        metrics[f'f1@{k}'] = f1
     
    return metrics

def soft_full_metrics(embeddings: torch.Tensor,
                      labels: torch.Tensor,
                      k_vals=(1, 5, 10),
                      theta: float = 0.5,
                      batch_size: int = 1000,
                      device='cuda'):
    """
    embeddings: [N, D], labels: [N]
    Return: dict of soft_recall@k and soft_precision@k over full set
    """
    with torch.no_grad():
        N, D = embeddings.shape
        emb = F.normalize(embeddings.to(device), p=2, dim=1)
        emb_T = emb.t()
        lbl = labels.to(device)

        metrics = {}
        for k in tqdm(k_vals,desc=f"Computing metrics@k",ascii=True):
            soft_rec, soft_prec = 0.0, 0.0
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch_emb = emb[start:end]  # [B,D]
                sim = batch_emb @ emb_T     # [B,N]
                sim[:, start:end].fill_diagonal_(-float('inf'))  # mask self

                _, topk = sim.topk(k, dim=1)  # [B,k]
                sims = torch.gather(sim, 1, topk)  # [B,k]

                hits = sims >= theta  # [B,k]
                soft_rec += hits.any(dim=1).float().sum()
                soft_prec += hits.sum() / k

            metrics[f'soft_recall@{k}'] = (soft_rec / N).item()
            metrics[f'soft_precision@{k}'] = (soft_prec / N).item()

        return metrics
