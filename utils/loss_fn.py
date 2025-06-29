import torch
import torch.nn.functional as F
import torch.nn as nn

class InfoNceLoss(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.reduction = reduction

    def forward(self, context_emb: torch.Tensor, gloss_emb: torch.Tensor, labels: torch.Tensor):        
        C = F.normalize(context_emb, p=2, dim=1)    # [N,D]
        G = F.normalize(gloss_emb, p=2, dim=1)      # [N,D]
        sim = torch.matmul(C, G.T) / self.temperature  # [N,N]
        
        N = sim.size(0)
        device = sim.device
        
        mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        mask_pos.fill_diagonal_(False)
        exp_sim = torch.exp(sim)
        sum_pos = (exp_sim * mask_pos.float()).sum(dim=1) + self.eps
        
        sum_all = (exp_sim * (~torch.eye(N, device=device).bool()).float()).sum(dim=1) + self.eps
        loss = - torch.log(sum_pos / sum_all)
        
        no_pos = (mask_pos.sum(dim=1) == 0)
        if no_pos.any():
            sim_no_diag = sim.masked_fill(torch.eye(N, device=device).bool(), -1e9)
            max_neg = torch.max(sim_no_diag, dim=1).values
            loss[no_pos] = - max_neg[no_pos]

        return loss.mean() if self.reduction=='mean' else loss.sum()
    
    
class DistillLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, context_emb: torch.Tensor, gloss_emb: torch.Tensor):
        # Normalize
        C = F.normalize(context_emb, p=2, dim=1)   
        G = F.normalize(gloss_emb, p=2, dim=1)
        
        dist_c = 1 - torch.matmul(C, C.T)
        dist_g = 1 - torch.matmul(G, G.T)
        return F.mse_loss(dist_c, dist_g, reduction=self.reduction)

class InfonceDistillLoss(nn.Module):
    def __init__(self,temperature=0.2, eps=1e-8, aux_weight = 0.5,
                 info_reduction='mean', distill_reduction: str = 'mean'):
        super().__init__()
        self.infonce_loss = InfoNceLoss(
            temperature=temperature,
            eps=eps,
            reduction=info_reduction
        )
        self.distill_loss = DistillLoss(
            reduction=distill_reduction
        )
        self.aux_weight = aux_weight
    
    def forward(self, context_emb: torch.Tensor, gloss_emb: torch.Tensor, labels: torch.Tensor):
        loss_nce = self.infonce_loss(context_emb, gloss_emb, labels)
        loss_dist = self.distill_loss(context_emb, gloss_emb)
        
        return loss_nce + loss_dist*self.aux_weight