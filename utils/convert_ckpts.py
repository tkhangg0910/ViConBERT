import os
import json
import torch
from transformers import PreTrainedTokenizerFast
from models.base_model import SynoViSenseEmbeddingV2, SynoViSenseEmbeddingV1
from utils.load_config import load_config
def convert_checkpoint(base_model,old_checkpoint_path, output_dir, tokenizer_name="vinai/phobert-base"):
    
    config = load_config("configs/stage1.yml")
    
    checkpoint = torch.load(old_checkpoint_path, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint  
    
    optional = {
        "context_window_size":config["model"]["context_window_size"]
    }
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    model = base_model(tokenizer,
                model_name=config["base_model"],
                cache_dir=config["base_model_cache_dir"],
                fusion_hidden_dim=config["model"]["fusion_hidden_dim"],
                dropout=config["model"]["dropout"],
                freeze_base=config["model"]["freeze_base"],
                fusion_num_layers=config["model"]["fusion_num_layers"],
                wp_num_layers=config["model"]["wp_num_layers"],
                cp_num_layers=config["model"]["cp_num_layers"],
                **optional)
   
    model.load_state_dict(state_dict)
    
    model.config = config
    
    model.tokenizer = tokenizer
    
    os.makedirs(output_dir, exist_ok=True)
    
    model.save_pretrained(output_dir)
    
    print(f"Đã chuyển đổi checkpoint thành công!")
    print(f"Đầu ra được lưu tại: {output_dir}")

if __name__ == "__main__":
    # Thay đổi các đường dẫn này theo thực tế
    convert_checkpoint(
        base_model=SynoViSenseEmbeddingV2,
        old_checkpoint_path="checkpoints/stage1/phobert_base/run_20250625_v1/best_model.pt",  
        output_dir="checkpoints/stage1/phobert_base/run_20250625_v1/converted_model",  # Thư mục đầu ra
        tokenizer_name="vinai/phobert-base"  # Tên tokenizer
    )