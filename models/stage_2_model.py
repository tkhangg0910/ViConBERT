from .base_model import SynoViSenseEmbeddingV2, MLPBlock

class SuperSensePredModel(SynoViSenseEmbeddingV2):
    def __init__(self, tokenizer, model_name: str = "vinai/phobert-base", 
                 cache_dir: str = "embeddings/base_models", fusion_hidden_dim: int = 512, wp_num_layers: int = 1, 
                 cp_num_layers: int = 1, dropout: float = 0.1, freeze_base: bool = False, fusion_num_layers: int = 1, 
                 context_window_size: int = 3,
                 supersense_size: int  = 47,
                 pred_head_num_layer:int= 1
                 ):
        super().__init__(tokenizer, model_name, cache_dir, 
                         fusion_hidden_dim, wp_num_layers, 
                         cp_num_layers, dropout, freeze_base, 
                         fusion_num_layers, context_window_size)
        
        prediction_head = MLPBlock(self.hidden_size,
                                   fusion_hidden_dim,
                                   supersense_size,
                                   pred_head_num_layer
                                   )