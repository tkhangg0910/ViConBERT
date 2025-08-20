import json
import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers.utils import is_torch_available
from transformers import PreTrainedTokenizerFast, PhobertTokenizerFast, XLMRobertaTokenizerFast, DebertaV2TokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

from data.processed.polybert_exp.dataset import PolyBERTtDataset, ContrastiveBatchSampler, PolyBERTtDataseV2,PolyBERTtDatasetV3
from models.polybert import PolyBERT
from utils.load_config import load_config
from utils.optimizer import create_optimizer
from trainings.train_polybert.utils import train_model
from trainings.train_polybert.utils_bc import train_model_bc

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--load_ckpts", action='store_true', help="Model type")
    # parser.add_argument("--model_type", type=str, default="base", help="Model type")
    parser.add_argument('--grad_clip', action='store_true', help='Gradient clipping')
    # parser.add_argument('--dataset_mode',  type=str, default='flat', help='dataset_mode')
    parser.add_argument('--grad_accum_steps',  type=int, default=1, help='dataset_mode')
    parser.add_argument('--train_gloss_size',  type=int, default=5, help='train_gloss_size')

    parser.add_argument('--train_mode',  type=str, default="bc", help='dataset_mode')

    args = parser.parse_args()
    return args 
        
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) 
    args = setup_args()
    print(f"Load From Checkpoint: {bool(args.load_ckpts)}")
    print(f"Device: {device}")
    print(f"grad_accum_steps: {args.grad_accum_steps}")
    config = load_config(f"configs/poly.yml")
    print(f"base_model: {config['base_model']}")
    
    with open(config["data"]["train_path"], "r",encoding="utf-8") as f:
        train_sample = json.load(f)
    with open(config["data"]["valid_path"], "r",encoding="utf-8") as f:
        valid_sample = json.load(f)

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
    if args.train_mode == "bc":
        train_dataset = PolyBERTtDatasetV3(train_sample, tokenizer )
        valid_dataset = PolyBERTtDatasetV3(valid_sample, tokenizer, val_mode=True)
        sampler = ContrastiveBatchSampler(train_dataset,batch_size=batch_size)

        train_dataloader = DataLoader(
            train_dataset,
            # batch_size=batch_size,
            collate_fn=train_dataset.collate_fn,
            batch_sampler=sampler,
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

    else:
        train_dataset = PolyBERTtDataseV2(train_sample, tokenizer, train_gloss_size= args.train_gloss_size )
        valid_dataset = PolyBERTtDataseV2(valid_sample, tokenizer,val_mode=True)
    
    # sampler = ContrastiveBatchSampler(train_dataset,batch_size=batch_size)
    
        train_dataloader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                # batch_sampler=sampler,
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
    
    if bool(args.load_ckpts):
        model = PolyBERT.from_pretrained(config["base_model"]).to(device)
    else:
        model = PolyBERT(
            polym = config["model"]["polym"],
            num_heads = config["model"]["num_heads"],
            bert_model_name=config["base_model"],
            tokenizer=tokenizer,
            ).to(device)
    total_steps = len(train_dataloader) * config["training"]["epochs"] 
    steps_per_epoch = len(train_dataloader)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    warmup_steps = int(0.1 * total_steps)
    
    optim = create_optimizer(model, config)

    scheduler = ReduceLROnPlateau(
        optimizer = optim,
        mode='min',         
        factor=0.5,         
        patience=2,         
        min_lr=1e-6          
    )
    Trainer = train_model_bc if args.train_mode == "bc" else train_model
    history, trained_model = Trainer(
        num_epochs=config["training"]["epochs"],
        train_data_loader=train_dataloader,
        valid_data_loader=valid_dataloader,
        optimizer=optim,
        model=model,
        device=device,
        checkpoint_dir=config["output_dir"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        ckpt_interval=config["training"]["ckpt_interval"],
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum_steps
    )
