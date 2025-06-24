import json
import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers.utils import is_torch_available
from transformers import PreTrainedTokenizerFast, PhobertTokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.processed.stage1_pseudo_sents.pseudo_sent_datasets import PseudoSents_Dataset
from models.base_model import SynoViSenseEmbeddingV1, SynoViSenseEmbeddingV2
from utils.load_config import load_config
from utils.optimizer import create_optimizer
from utils.loss_fn import InfoNceLoss
from trainings.utils import train_model

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--model", type=str, default="v2", help="Model type")
    args = parser.parse_args()
    return args 
        
if __name__=="__main__":
    args = setup_args()

    config = load_config("configs/stage1.yml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    with open(config["data"]["train_path"], "r",encoding="utf-8") as f:
        train_sample = json.load(f)
    with open(config["data"]["valid_path"], "r",encoding="utf-8") as f:
        valid_sample = json.load(f)
    
    if args.model=="v2":              
        tokenizer = PreTrainedTokenizerFast.from_pretrained(config["base_model"])
    else:
        tokenizer = PhobertTokenizerFast.from_pretrained(config["base_model"])
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    
    train_set = PseudoSents_Dataset(train_sample, tokenizer
                                    , is_training=True, 
                                    num_synsets_per_batch=128,samples_per_synset=6,
                                    use_sent_masking=args.model=="v1")
    valid_set = PseudoSents_Dataset(valid_sample, tokenizer, is_training=False
                                    ,use_sent_masking=args.model=="v1")
    
    # sampler = train_set.get_weighted_sampler()

    # custom_batch_sampler = CustomSynsetAwareBatchSampler(
    #     train_set, sampler=sampler, batch_size=config["training"]["batch_size"], drop_last=False
    # )   

    train_dataloader = DataLoader(train_set,
                                  config["training"]["batch_size"],
                                  shuffle=False,
                                  collate_fn=train_set.custom_collate_fn,
                                  num_workers=config["data"]["num_workers"],
                                  pin_memory=True
                                  )
    valid_dataloader = DataLoader(valid_set,
                                  config["training"]["batch_size"],
                                  shuffle=False,
                                  collate_fn=valid_set.custom_collate_fn,
                                  num_workers=config["data"]["num_workers"],
                                  pin_memory=True
                                  )
    
    arc = SynoViSenseEmbeddingV1 if args.model=="v1" else SynoViSenseEmbeddingV2
    optional={
        "context_window_size":config["model"]["context_window_size"]
        }if args.model=="v2" else {}
    model = arc(tokenizer,
                model_name=config["base_model"],
                cache_dir=config["base_model_cache_dir"],
                fusion_hidden_dim=config["model"]["fusion_hidden_dim"],
                dropout=config["model"]["dropout"],
                freeze_base=config["model"]["freeze_base"],
                fusion_num_layers=config["model"]["fusion_num_layers"],
                wp_num_layers=config["model"]["wp_num_layers"],
                cp_num_layers=config["model"]["cp_num_layers"],
                **optional
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

    
    loss_fn = InfoNceLoss()

    history, trained_model = train_model(
        num_epochs=config["training"]["epochs"],
        train_data_loader=train_dataloader,
        valid_data_loader=valid_dataloader,
        loss_fn=loss_fn,
        optimizer=optim,
        model=model,
        device=device,
        checkpoint_dir=config["output_dir"],
        early_stopping_patience=config["training"]["early_stopping_patience"],
        ckpt_interval=config["training"]["ckpt_interval"]
    )
