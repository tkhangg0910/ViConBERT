import torch
import torch.nn.functional as F

def stage_1_supcon_loss(features, labels, temp=0.1):
    device = features.device
    batch_size = features.shape[0]
    
    if torch.isnan(features).any() or torch.isinf(features).any():
        return torch.tensor(0.0, device=device)

    features = F.normalize(features, p=2, dim=1)


    similarity_matrix = torch.matmul(features, features.T) / temp
    
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)
    
    mask = mask * (1 - torch.eye(batch_size, device=device))  
    
    logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)

    logits = similarity_matrix - logits_max.detach()

    exp_logits = torch.exp(logits)

    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)

    mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask.sum(dim=1) + 1e-8)

    
    loss = -mean_log_prob_pos.mean()

    return loss
