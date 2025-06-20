import torch
import torch.nn.functional as F
import torch.nn as nn

class InfoNceLoss(nn.Module):
    def __init__(self,temperature=0.1):
        self.temperature = temperature
        
    def forward(self,embeddings, labels):
        """
        embeddings: Normalized embeddings [batch_size, dim]
        labels: Temporary synset labels [batch_size] (0-31 per batch)
        temperature: Softmax temperature
        """
        sim_matrix = embeddings @ embeddings.T  # [N, N]
        
        same_group = labels.unsqueeze(0) == labels.unsqueeze(1)  # [N, N]
        same_group.fill_diagonal_(False) 
        
        pos_sim = torch.exp(sim_matrix / self.temperature) * same_group
        numerator = pos_sim.sum(dim=1)  # [N]
        
        exp_sim = torch.exp(sim_matrix / self.temperature)
        denominator = exp_sim.sum(dim=1) - torch.exp(torch.ones(embeddings.size(0)) / self.temperature)
        
        loss_per_sample = -torch.log(numerator / denominator)
        return loss_per_sample.mean()
