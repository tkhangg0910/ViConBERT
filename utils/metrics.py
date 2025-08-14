import torch
import numpy as np
from sklearn.metrics import normalized_mutual_info_score
from sklearn.cluster import KMeans
from tqdm import tqdm
import torch.nn.functional as F
import faiss

def compute_precision_recall_f1_for_wsd(MF, word_id):
    """
    MF: [B, B] similarity matrix (context vs gloss embeddings)
    word_id: [B] global word ids
    """
    device = MF.device
    B = MF.size(0)
    
    # positive mask: same global word_id
    word_id_row = word_id.unsqueeze(0)   # [1, B]
    word_id_col = word_id.unsqueeze(1)   # [B, 1]
    pos_mask = (word_id_row == word_id_col).float()  # [B, B]

    # predicted gloss: index of max similarity for each context
    pred_idx = MF.argmax(dim=1)  # [B]

    # TP / FP / FN
    tp = 0
    fp = 0
    fn = 0
    for i in range(B):
        # positive indices for this sample
        pos_indices = torch.nonzero(pos_mask[i]).flatten().tolist()
        if pred_idx[i].item() in pos_indices:
            tp += 1
        else:
            fp += 1
            fn += 1  # missed positive

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    
    return precision, recall, f1

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

def compute_step_metrics(embeddings, labels, k_vals=(1, 5, 10), device='cuda',ndcg=True ):
    """
    Quickly compute evaluation metrics for a training step
    Returns a dictionary of computed metrics
    """
    metrics = {}

    for k in k_vals:
        metrics[f'recall@{k}'] = recall_at_k_batch(embeddings, labels, k, device=device)
        metrics[f'precision@{k}'] = precision_at_k_batch(embeddings, labels, k, device=device)
        if ndcg:
            metrics[f'ndcg@{k}'] = ndcg_at_k_batch(embeddings, labels, k, device=device)
    
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

def ndcg_at_k_batch(preds: torch.Tensor, labels: torch.Tensor, k: int, device='cuda'):
    """
    preds: [B, D] - predicted embeddings
    labels: [B]   - true labels
    Return: average nDCG@k over the batch
    """
    B = preds.size(0)
    preds = F.normalize(preds, p=2, dim=1).to(device)
    labels = labels.to(device)
    
    sim = torch.matmul(preds, preds.T)          # [B, B]
    diag = torch.eye(B, dtype=torch.bool, device=device)
    sim.masked_fill_(diag, -float('inf'))       # remove self-similarity

    _, topk_indices = sim.topk(k, dim=1)        # [B, k]
    gains = torch.zeros((B, k), device=device)

    for i in range(B):
        hits = (labels[topk_indices[i]] == labels[i]).float()
        gains[i] = hits / torch.log2(torch.arange(2, k + 2, device=device).float())

    dcg = gains.sum(dim=1)
    ideal_gains = torch.ones_like(gains)
    ideal_dcg = ideal_gains / torch.log2(torch.arange(2, k + 2, device=device).float())
    idcg = ideal_dcg.sum(dim=1)

    ndcg = dcg / idcg
    return ndcg.mean().item()

def compute_full_ndcg_large_scale(embeddings, labels, k_vals=(1, 5, 10), device='cuda'):
    """
    embeddings: [N, D], labels: [N]
    """
    embeddings = F.normalize(embeddings.to(device), p=2, dim=1)
    labels = labels.to(device)
    N = embeddings.size(0)
    ndcg_scores = {f'ndcg@{k}': 0.0 for k in k_vals}
    
    for i in tqdm(range(N), desc="Computing full nDCG", ascii=True):
        query = embeddings[i].unsqueeze(0)           # [1, D]
        sim = torch.matmul(query, embeddings.T).squeeze(0)  # [N]
        sim[i] = -float('inf')                        # mask self

        sorted_indices = torch.topk(sim, k=max(k_vals)).indices  # [max_k]
        for k in k_vals:
            topk = sorted_indices[:k]
            hits = (labels[topk] == labels[i]).float()
            gain = hits / torch.log2(torch.arange(2, k + 2, device=device).float())
            dcg = gain.sum()
            ideal_gain = torch.ones_like(hits) / torch.log2(torch.arange(2, k + 2, device=device).float())
            idcg = ideal_gain.sum()
            ndcg_scores[f'ndcg@{k}'] += (dcg / idcg).item()

    return {k: v / N for k, v in ndcg_scores.items()}




def compute_ndcg_from_faiss(
    context_embd: torch.Tensor,
    true_synset_labels: torch.Tensor, 
    faiss_index_path: str,
    synset_id_map_path: str, 
    label_to_synset_map: dict, 
    k_vals=(1, 5, 10)
):
    """
    context_embd: [N, D] tensor (can be on GPU or CPU)
    true_synset_labels: [N] tensor, int labels
    faiss_index_path: path to .index file
    synset_id_map_path: path to FAISS index → raw synset id mapping
    label_to_synset_map: maps true label (int) → raw synset_id (str)

    Returns:
        dict: { 'ndcg@1': ..., 'ndcg@5': ..., 'ndcg@10': ... }
    """
    context_embd = F.normalize(context_embd, p=2, dim=1).cpu().numpy()
    true_synset_labels = true_synset_labels.cpu().numpy()

    # Load FAISS index and index → raw synset ID mapping
    index = faiss.read_index(faiss_index_path)
    synset_id_map = torch.load(synset_id_map_path)  # dict: faiss_idx → raw_synset_id (str)

    max_k = max(k_vals)
    sim, indices = index.search(context_embd, max_k)

    ndcg_scores = {f'ndcg@{k}': 0.0 for k in k_vals}
    N = len(context_embd)

    for i in tqdm(range(N),desc="Computing nDCG", ascii=True):
        true_synset_id = label_to_synset_map[true_synset_labels[i]]
        retrieved_synset_ids = [synset_id_map[j] for j in indices[i]]

        for k in k_vals:
            topk_ids = retrieved_synset_ids[:k]
            hits = np.array([1.0 if sid == true_synset_id else 0.0 for sid in topk_ids], dtype=np.float32)
            gains = hits / np.log2(np.arange(2, k + 2))
            dcg = gains.sum()
            idcg = (np.ones(int(hits.sum())) / np.log2(np.arange(2, int(hits.sum()) + 2))).sum() if hits.sum() > 0 else 1.0
            ndcg = dcg / idcg
            ndcg_scores[f'ndcg@{k}'] += ndcg

    return {k: v / N for k, v in ndcg_scores.items()}
