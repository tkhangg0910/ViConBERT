import torch
import torch.nn.functional as F
import torch.nn as nn

class InfoNceLoss(nn.Module):
    def __init__(self, temperature=0.5, eps=1e-8):  
        super(InfoNceLoss, self).__init__()  
        self.temperature = temperature
        self.eps = eps  
        
    def forward(self, embeddings, labels):
        """
        embeddings: Should be L2 normalized embeddings [batch_size, dim]
        labels: Temporary synset labels [batch_size] (0-31 per batch)
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        sim_matrix = embeddings @ embeddings.T / self.temperature
        
        same_group = labels.unsqueeze(0) == labels.unsqueeze(1)
        same_group.fill_diagonal_(False)
        
        pos_counts = same_group.sum(dim=1)
        if (pos_counts == 0).any():
            print("No Pos")
            valid_mask = pos_counts > 0
            if valid_mask.sum() == 0:
                return torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
            sim_matrix = sim_matrix[valid_mask][:, valid_mask]
            same_group = same_group[valid_mask][:, valid_mask]
            

        sim_max = sim_matrix.max(dim=1, keepdim=True)[0].detach()
        sim_matrix_stable = sim_matrix - sim_max
        
        exp_sim = torch.exp(sim_matrix_stable)
        
        pos_exp_sim = exp_sim * same_group
        pos_sum = pos_exp_sim.sum(dim=1)
        
        mask_not_self = ~torch.eye(exp_sim.size(0), dtype=torch.bool, device=exp_sim.device)
        all_sum = (exp_sim * mask_not_self).sum(dim=1)
        
        pos_sum = pos_sum + self.eps
        all_sum = all_sum + self.eps
        
        loss = -torch.log(pos_sum / all_sum)
        
        return loss.mean()
