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
        dtype = context_emb.dtype

        C = F.normalize(context_emb, p=2, dim=1)    # [N, PXD]
        G = F.normalize(gloss_emb, p=2, dim=1)      # [N,PXD]
        sim = torch.matmul(C, G.T) / self.temperature  # [N,N]
        sim = sim.to(dtype)  
        N = sim.size(1)
        device = sim.device
        
        mask_pos = labels.unsqueeze(1).eq(labels.unsqueeze(0))
        mask_pos.fill_diagonal_(False)
        exp_sim = torch.exp(sim)
        sum_pos = (exp_sim * mask_pos.to(dtype)).sum(dim=1) + self.eps
        sum_all = (exp_sim * (~torch.eye(N, device=device).bool()).to(dtype)).sum(dim=1) + self.eps

        frac = (sum_pos / sum_all).clamp(min=self.eps, max=1.0)

        loss = - torch.log(frac)
        
        no_pos = (mask_pos.sum(dim=1) == 0)
        if no_pos.any():
            sim_no_diag = sim.masked_fill(torch.eye(N, device=device).bool(), -float("inf"))
            max_neg = torch.max(sim_no_diag, dim=1).values
            loss[no_pos] = (-max_neg[no_pos]).to(loss.dtype)  

        return loss.mean() if self.reduction=='mean' else loss.sum()

class InfoNceLossV2(nn.Module):
    def __init__(self, temperature=0.1, eps=1e-8, reduction='mean'):
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.reduction = reduction

    def forward(self, context_emb: torch.Tensor,
                      gloss_emb: torch.Tensor,
                      labels: torch.Tensor):
        # normalize
        C = F.normalize(context_emb, p=2, dim=1)
        G = F.normalize(gloss_emb,  p=2, dim=1)
        # [N,N] similarity
        sim = (C @ G.t()) / self.temperature

        N = sim.size(0)
        device = sim.device

        # exp
        exp_sim = torch.exp(sim)

        # mask out self‐similarities on the diagonal
        mask_eye = torch.eye(N, dtype=torch.bool, device=device)
        exp_sim.masked_fill_(mask_eye, 0.0)

        # positive mask: same label but not self
        mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
        mask_pos = mask_pos & (~mask_eye)

        # sum of positives, sum of all (=> positives + negatives)
        sum_pos = (exp_sim * mask_pos).sum(dim=1)         # [N]
        sum_all = exp_sim.sum(dim=1)                     # [N]

        # frac in [0,1]
        frac = sum_pos / (sum_all + self.eps)
        # clamp for numeric safety
        frac = frac.clamp(min=self.eps, max=1.0)

        # loss = - log(frac)
        loss = -torch.log(frac)

        # for samples with no positives: fallback to hardest negative
        no_pos = (mask_pos.sum(dim=1) == 0)
        if no_pos.any():
            # set their loss = max_neg_score * (-1)
            sim_neg = sim.masked_fill(mask_eye, -float('inf'))
            hardest_neg = torch.max(sim_neg, dim=1).values
            loss[no_pos] = (-hardest_neg[no_pos]).clamp(min=0)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    
class DistillLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, context_emb: torch.Tensor, gloss_emb: torch.Tensor):
        dtype = context_emb.dtype

        C = F.normalize(context_emb, p=2, dim=1).to(dtype)
        G = F.normalize(gloss_emb, p=2, dim=1).to(dtype)

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
    

class CosinMarginDistillLoss(nn.Module):
    def __init__(self,margin= 0.5, aux_weight = 0.5,
                 cosin_reduction='mean', distill_reduction: str = 'mean'):
        super().__init__()
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin, reduction=cosin_reduction)
        self.distill_loss = DistillLoss(
            reduction=distill_reduction
        )
        self.aux_weight = aux_weight
    
    def forward(self, context_emb: torch.Tensor, gloss_emb: torch.Tensor, labels: torch.Tensor):
        loss_nce = self.cosine_loss(context_emb, gloss_emb, labels)
        loss_dist = self.distill_loss(context_emb, gloss_emb)
        
        return loss_nce + loss_dist*self.aux_weight