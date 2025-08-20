import json
import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers.utils import is_torch_available
from transformers import PreTrainedTokenizerFast, PhobertTokenizerFast, XLMRobertaTokenizerFast, DebertaV2TokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

from data.processed.bem_exp.dataset import BEMDataset  # Updated dataset with mask extractor
from models.bem import BiEncoderModel
from utils.load_config import load_config
from utils.optimizer import create_optimizer
from trainings.train_bem.utils import train_model

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--load_ckpts", action='store_true', help="Model type")
    parser.add_argument('--grad_clip', action='store_true', help='Gradient clipping')
    parser.add_argument('--grad_accum_steps',  type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--train_gloss_size',  type=int, default=5, help='Number of glosses for training')
    parser.add_argument('--train_mode',  type=str, default="bc", help='Training mode')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    return args 
        
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) 
    args = setup_args()
    
    print(f"Load From Checkpoint: {bool(args.load_ckpts)}")
    print(f"Device: {device}")
    print(f"grad_accum_steps: {args.grad_accum_steps}")
    print(f"train_gloss_size: {args.train_gloss_size}")
    print(f"Debug mode: {args.debug}")
    
    config = load_config(f"configs/bem.yml")
    print(f"base_model: {config['base_model']}")
    
    with open(config["data"]["train_path"], "r", encoding="utf-8") as f:
        train_sample = json.load(f)
    with open(config["data"]["valid_path"], "r", encoding="utf-8") as f:
        valid_sample = json.load(f)

    # Initialize tokenizer based on model type
    if config["base_model"].startswith("vinai"):
        print("using PhobertTokenizerFast")
        tokenizer = PhobertTokenizerFast.from_pretrained(config["base_model"])
    elif config["base_model"].startswith("FacebookAI"):
        print("using XLMRobertaTokenizerFast")
        tokenizer = XLMRobertaTokenizerFast.from_pretrained(config["base_model"])
    elif config["base_model"].startswith("Fsoft-AIC"):
        print("using DebertaTokenizerFast")
        tokenizer = DebertaV2TokenizerFast.from_pretrained(config["base_model"])
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    batch_size = config["training"]['batch_size']
    
    # Initialize datasets with mask extractor
    print("Initializing training dataset...")
    train_dataset = BEMDataset(train_sample, tokenizer, train_gloss_size=args.train_gloss_size)
    print("Initializing validation dataset...")
    valid_dataset = BEMDataset(valid_sample, tokenizer, val_mode=True)
    
    # Debug: Check a few samples
    if args.debug:
        print("\n=== Debug Information ===")
        for i in range(min(3, len(train_dataset))):
            debug_info = train_dataset.get_sample_for_debug(i)
            print(f"\nSample {i}:")
            print(f"Sentence: {debug_info['sentence']}")
            print(f"Target: {debug_info['target_word']}")
            print(f"Mask sum: {sum(debug_info['target_mask'])}")
            print("Mask visualization:")
            print(debug_info['mask_visualization'])
            print("-" * 50)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=valid_dataset.collate_fn,
        num_workers=config["data"]["num_workers"],
        pin_memory=True
    )
    
    # Initialize model
    if bool(args.load_ckpts):
        model = BiEncoderModel.from_pretrained(config["base_model"]).to(device)
    else:
        model = BiEncoderModel(
            encoder_name=config["base_model"],
            tokenizer=tokenizer,
        ).to(device)
    
    total_steps = len(train_dataloader) * config["training"]["epochs"] 
    steps_per_epoch = len(train_dataloader)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    warmup_steps = int(0.1 * total_steps)
    
    optim = create_optimizer(model, config)
    
    scheduler = ReduceLROnPlateau(
        optimizer=optim,
        mode='min',         
        factor=0.5,         
        patience=2,         
        min_lr=1e-6          
    )

    # Test a batch to ensure everything works
    if args.debug:
        print("\n=== Testing batch processing ===")
        test_batch = next(iter(train_dataloader))
        print(f"Batch keys: {test_batch.keys()}")
        print(f"context_input_ids shape: {test_batch['context_input_ids'].shape}")
        print(f"context_attn_mask shape: {test_batch['context_attn_mask'].shape}")
        print(f"target_masks shape: {test_batch['target_masks'].shape}")
        print(f"Number of candidate_glosses: {len(test_batch['candidate_glosses'])}")
        print(f"gold_indices shape: {test_batch['gold_indices'].shape}")
        
        # Test forward pass
        try:
            with torch.no_grad():
                model.eval()
                ctx_vecs = model.forward_context(
                    test_batch['context_input_ids'][:2].to(device),
                    test_batch['context_attn_mask'][:2].to(device),
                    test_batch['target_masks'][:2].to(device)
                )
                print(f"Context vectors shape: {ctx_vecs.shape}")
                print("✓ Forward pass successful!")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")
            raise

    print("\n=== Starting Training ===")
    history, trained_model = train_model(
        num_epochs=config["training"]["epochs"],
        train_data_loader=train_dataloader,
        valid_data_loader=valid_dataloader,
        optimizer=optim,
        model=model,
        device=device,
        checkpoint_dir=config["output_dir"],
        scheduler=scheduler,
        early_stopping_patience=config["training"]["early_stopping_patience"],
        ckpt_interval=config["training"]["ckpt_interval"],
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps
    )
    
    print("Training completed!")