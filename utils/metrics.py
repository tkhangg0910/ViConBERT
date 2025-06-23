import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from tqdm import tqdm

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

def compute_full_metrics(embeddings, labels, k_vals=(1, 5, 10), device='cuda'):
    """
    Compute full evaluation metrics
    Returns a dictionary of computed metrics
    """
    metrics = {}
    
    for k in k_vals:
        metrics[f'recall@{k}'] = recall_at_k_full(embeddings, labels, k, device=device)
        metrics[f'precision@{k}'] = precision_at_k_full(embeddings, labels, k, device=device)
    
    # metrics['nmi'] = normalized_mutual_info(embeddings, labels)
    return metrics

def compute_full_metrics_large_scale(embeddings, labels, k_vals, device='cuda'):
    """
    Computes metrics for very large datasets using efficient chunking
    Optimized for validation set with 66k+ samples
    """
    # Normalize embeddings for cosine similarity
    embeddings = normalize_embeddings(embeddings)
    n = len(embeddings)
    
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
    
    print(f"Using chunk size: {chunk_size} for validation metrics")
    
    max_k = max(k_vals)
    
    # Storage for top-k results
    topk_indices = torch.empty((n, max_k), dtype=torch.long, device='cpu')
    
    # Compute similarity in chunks
    for i in tqdm(range(0, n, chunk_size), desc="Computing similarity"):
        start_i = i
        end_i = min(i + chunk_size, n)
        chunk_size_actual = end_i - start_i
        
        # Compute chunk similarity
        chunk = embeddings[start_i:end_i].to(device)
        full_emb = embeddings.to(device)
        sim = torch.mm(chunk, full_emb.t())
        
        # Set diagonal to -inf to exclude self-similarity
        for j in range(chunk_size_actual):
            idx = start_i + j
            sim[j, idx] = -float('inf')
        
        # Get top-k for this chunk
        _, chunk_topk_indices = torch.topk(sim, max_k, dim=1)
        topk_indices[start_i:end_i] = chunk_topk_indices.cpu()
    
    # Move labels to CPU for efficient computation
    labels = labels.cpu()
    
    # Calculate metrics
    metrics = {}
    for k in k_vals:
        recall_sum = 0.0
        precision_sum = 0.0
        
        for i in tqdm(range(n), desc=f"Computing metrics@k={k}"):
            # Get top-k indices for this sample
            topk = topk_indices[i, :k]
            
            # Find matches
            matches = (labels[topk] == labels[i])
            
            # Recall: at least one match in top-k
            recall_sum += matches.any().item()
            
            # Precision: fraction of matches in top-k
            precision_sum += matches.float().mean().item()
        
        metrics[f'recall@{k}'] = recall_sum / n
        metrics[f'precision@{k}'] = precision_sum / n
    
    # Compute NMI with sampling
    # metrics['nmi'] = normalized_mutual_info(embeddings.cpu(), labels.cpu())
    
    return metrics
