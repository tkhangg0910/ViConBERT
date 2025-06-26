import os
import torch
import torch.nn as nn
from .base_model import SynoViSenseEmbeddingV2, MLPBlock

class SuperSensePredModel(nn.Module):
    def __init__(self,
                 embedding_model: SynoViSenseEmbeddingV2, 
                 supersense_size: int = 47,
                 pred_head_num_layer: int = 1,
                 prediction_hidden_dim: int = 512
                 ):
        super().__init__()
        self.supersense_size =supersense_size
        self.pred_head_num_layer= pred_head_num_layer
        self.prediction_hidden_dim = prediction_hidden_dim
        self.embedding_model = embedding_model
        self.prediction_head = MLPBlock(embedding_model.hidden_size,
                                   prediction_hidden_dim,
                                   supersense_size,
                                   pred_head_num_layer
                                   )
    def forward(self, 
            word_input_ids: torch.Tensor,
            word_attention_mask: torch.Tensor,
            context_input_ids: torch.Tensor,
            context_attention_mask: torch.Tensor,
            target_spans: torch.Tensor):
    
        fused_embed = self.embedding_model(
            word_input_ids,
            word_attention_mask,
            context_input_ids,
            context_attention_mask,
            target_spans
        )
    
        return self.prediction_head(fused_embed)

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        
        embedding_dir = os.path.join(save_directory, "embedding_model")
        self.embedding_model.save_pretrained(embedding_dir)
        
        head_state = {
            'prediction_head': self.prediction_head.state_dict(),
            'config': {
                'supersense_size': self.supersense_size,
                'pred_head_num_layer': self.pred_head_num_layer,
                'fusion_hidden_dim': self.prediction_hidden_dim
            }
        }
        torch.save(head_state, os.path.join(save_directory, "prediction_head.bin"))
    
    @classmethod
    def from_pretrained(cls, save_directory, tokenizer=None):
        embedding_dir = os.path.join(save_directory, "embedding_model")
        embedding_model = SynoViSenseEmbeddingV2.from_pretrained(embedding_dir, tokenizer)
        
        head_state = torch.load(
            os.path.join(save_directory, "prediction_head.bin"),
            map_location=torch.device('cpu')
        )
        
        model = cls(
            embedding_model=embedding_model,
            supersense_size=head_state['config']['supersense_size'],
            pred_head_num_layer=head_state['config']['pred_head_num_layer'],
            prediction_hidden_dim=head_state['config']['prediction_hidden_dim']
        )
        
        model.prediction_head.load_state_dict(head_state['prediction_head'])
        return model

        
