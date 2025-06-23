import torch
import torch.nn.functional as F
import torch.nn as nn

class InfoNceLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.reduction = reduction

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = embeddings @ embeddings.T / self.temperature
        
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        pos_mask = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        for label in torch.unique(labels):
            indices = torch.where(labels == label)[0]
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(len(indices)):
                        if i != j:
                            pos_mask[indices[i], indices[j]] = True
        
        exp_sim = torch.exp(sim_matrix)
        
        pos_sum = (exp_sim * pos_mask).sum(dim=1) + self.eps
        
        all_sum = exp_sim.sum(dim=1) - torch.exp(torch.diag(sim_matrix)) + self.eps
        
        log_pos = torch.log(pos_sum)
        log_all = torch.log(all_sum)
        losses = -(log_pos - log_all)
        
        no_pos_mask = (pos_mask.sum(dim=1) == 0)
        if no_pos_mask.any():
            max_neg = torch.max(sim_matrix.masked_fill(pos_mask, -float('inf')), dim=1)[0]
            losses[no_pos_mask] = -max_neg[no_pos_mask]  # Đẩy xa negative nhất
            
        if self.reduction == 'mean':
            return losses.mean()
        elif self.reduction == 'sum':
            return losses.sum()
        else:
            return losses
