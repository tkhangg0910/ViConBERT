import torch
import torch.nn.functional as F

def stage_1_supcon_loss(features, labels, temp=0.1):
    device = features.device
    batch_size = features.shape[0]
    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T) / temp
    
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)

    self_mask = torch.eye(batch_size, dtype=torch.float, device=device)
    
    positive_mask = mask - self_mask

    exp_sim = torch.exp(similarity_matrix)
    log_prob = similarity_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    
    loss = - (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1)
    return loss.mean()
