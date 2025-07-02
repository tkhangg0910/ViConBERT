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
    def __init__(self, temperature: float = 0.1, eps: float = 1e-8, reduction: str = 'mean'):
        """
        Multi-positive InfoNCE loss: cho phép mỗi query có nhiều positives trong batch.
        
        Args:
            temperature: hệ số chia logits để điều chỉnh độ sắc nét.
            eps: tránh chia 0.
            reduction: 'mean' hoặc 'sum'.
        """
        super().__init__()
        self.temperature = temperature
        self.eps = eps
        self.reduction = reduction

    def forward(self, context_emb: torch.Tensor, gloss_emb: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            context_emb: Tensor [N, D]
            gloss_emb:    Tensor [N, D]  (thứ tự gloss tương ứng với context)
            labels:       LongTensor [N], nhãn synset cho mỗi cặp
        """
        # 1) Chuẩn hoá L2
        C = F.normalize(context_emb, p=2, dim=1)
        G = F.normalize(gloss_emb,    p=2, dim=1)

        # 2) Ma trận cosine logits [N,N]
        logits = torch.matmul(C, G.t()) / self.temperature

        N = logits.size(0)
        device = logits.device
        dtype = logits.dtype

        # 3) Tạo mask để exclude chính nó
        diag_mask = torch.eye(N, device=device, dtype=torch.bool)

        # 4) Tạo mask positives: P[i,j]=True nếu labels[i]==labels[j] và i!=j
        labels_i = labels.unsqueeze(1)          # [N,1]
        labels_j = labels.unsqueeze(0)          # [1,N]
        pos_mask = (labels_i == labels_j) & (~diag_mask)  # [N,N]

        # 5) exponential
        exp_logits = torch.exp(logits)  # [N,N]

        # 6) sum over positives và sum over tất cả neg (i!=i)
        sum_pos = (exp_logits * pos_mask.to(dtype)).sum(dim=1) + self.eps
        sum_all = (exp_logits * (~diag_mask).to(dtype)).sum(dim=1) + self.eps

        # 7) loss per sample và tổng hợp
        loss = - torch.log(sum_pos / sum_all)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss  # none

    
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