import torch
import torch.nn.functional as F
import torch.nn as nn

class stage_1_supcon_loss(nn.Module):
    def __init__(self, temp=0.1, margin=0.5):
        super().__init__()
        self.temp = temp
        self.margin = margin
        
    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        
        # Chuẩn hóa features
        features = F.normalize(features, p=2, dim=1)
        
        # Tính similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temp
        
        # Tạo mask cho positive pairs
        label_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        positive_mask = label_mask - torch.eye(batch_size, device=device)
        
        # Tìm hardest negatives
        negative_mask = 1 - label_mask
        negatives = sim_matrix * negative_mask
        hardest_negatives, _ = negatives.max(dim=1, keepdim=True)
        
        # Modified SupCon loss với hard negative mining
        numerator = torch.exp(sim_matrix) * positive_mask
        denominator = torch.exp(sim_matrix) + torch.exp(hardest_negatives + self.margin)
        
        log_prob = torch.log(numerator.sum(dim=1, keepdim=True) / denominator.sum(dim=1, keepdim=True))
        loss = -log_prob.mean()
        
        return loss

