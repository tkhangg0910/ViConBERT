import json
import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers.utils import is_torch_available
from transformers import PreTrainedTokenizerFast, PhobertTokenizerFast, XLMRobertaTokenizerFast, DebertaV2TokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from sentence_transformers import SentenceTransformer
import pandas as pd

from data.processed.stage1_pseudo_sents.pseudo_sent_datasets import PseudoSents_Dataset, PseudoSentsFlatDataset, SynsetBatchSampler
from models.polybert import PolyBERT
from utils.load_config import load_config
from utils.optimizer import create_optimizer
from trainings.train_polybert.utils import train_model

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--load_ckpts", action='store_true', help="Model type")
    parser.add_argument("--model_type", type=str, default="base", help="Model type")
    parser.add_argument("--only_multiple_el", action='store_true', help="Model type")
    parser.add_argument('--grad_clip', action='store_true', help='Gradient clipping')
    parser.add_argument('--dataset_mode',  type=str, default='flat', help='dataset_mode')
    parser.add_argument('--grad_accum_steps',  type=int, default=1, help='dataset_mode')
    args = parser.parse_args()
    return args 
        
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42) 
    args = setup_args()
    print(f"Load From Checkpoint: {bool(args.load_ckpts)}")
    print(f"only_multiple_el: {bool(args.only_multiple_el)}")
    print(f"Device: {device}")
    print(f"grad_accum_steps: {args.grad_accum_steps}")
    config = load_config(f"configs/{args.model_type}.yml")
    print(f"base_model: {config['base_model']}")
    print(f'Num head: {config["model"]["num_head"]}')
    print(f'Num Layers: {config["model"]["num_layers"]}')
    
    with open(config["data"]["train_path"], "r",encoding="utf-8") as f:
        train_sample = json.load(f)
    with open(config["data"]["valid_path"], "r",encoding="utf-8") as f:
        valid_sample = json.load(f)
            
    # gloss_enc = SentenceTransformer('dangvantuan/vietnamese-embedding'
                                    # ,cache_folder="embeddings/vietnamese_embedding")
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
    if args.dataset_mode == "sampling":
        train_set = PseudoSents_Dataset(config["data"]["emd_path"],
                                        # gloss_enc,
                                        train_sample, tokenizer
                                        , is_training=True, 
                                        num_synsets_per_batch=96,samples_per_synset=4,
                                        only_multiple_el=bool(args.only_multiple_el))

        valid_set = PseudoSents_Dataset(config["data"]["emd_path"],
                                        # gloss_enc,
                                        valid_sample, tokenizer, is_training=False
                                        ,only_multiple_el=bool(args.only_multiple_el))
        
        train_dataloader = DataLoader(train_set,
                                    1,
                                    shuffle=False,
                                    collate_fn=train_set.custom_collate_fn,
                                    num_workers=config["data"]["num_workers"],
                                    pin_memory=True
                                    )
        valid_dataloader = DataLoader(valid_set,
                                    1,
                                    shuffle=False,
                                    collate_fn=valid_set.custom_collate_fn,
                                    num_workers=config["data"]["num_workers"],
                                    pin_memory=True
                                    )
    elif args.dataset_mode == "flat":
        train_set = PseudoSentsFlatDataset(config["data"]["emd_path"],
                                        # gloss_enc,
                                        train_sample, tokenizer 
                                        )
        
        valid_set = PseudoSentsFlatDataset(config["data"]["emd_path"],
                                        # gloss_enc,
                                        valid_sample, tokenizer)
        batch_size = config["training"]["batch_size"]
        labels = [ train_set[i]["synset_ids"] for i in range(len(train_set)) ]

        sampler = SynsetBatchSampler(labels, batch_size,shuffle=True)
        
        train_dataloader = DataLoader(
            train_set,
            batch_sampler=sampler,
            collate_fn=train_set.collate_fn,
            num_workers=config["data"]["num_workers"],
            pin_memory=True
        )
        valid_dataloader = DataLoader(
            valid_set,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=valid_set.collate_fn,
            num_workers=config["data"]["num_workers"],
            pin_memory=True
        )

    if bool(args.load_ckpts):
        model = PolyBERT.from_pretrained(config["base_model"]).to(device)
    else:
        model = PolyBERT(
            tokenizer=tokenizer,
            ).to(device)
    
    total_steps = len(train_dataloader) * config["training"]["epochs"] 
    steps_per_epoch = len(train_dataloader)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    warmup_steps = int(0.1 * total_steps)
    
    optim = create_optimizer(model, config)
    
    # scheduler = get_linear_schedule_with_warmup(
    #     optim,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )
    scheduler = ReduceLROnPlateau(
        optimizer = optim,
        mode='min',         
        factor=0.5,         
        patience=2,         
        min_lr=1e-6          
    )

    

    history, trained_model = train_model(
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
