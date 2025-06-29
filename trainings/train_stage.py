import json
import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers.utils import is_torch_available
from transformers import PreTrainedTokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sentence_transformers import SentenceTransformer
import pandas as pd

from data.processed.stage1_pseudo_sents.pseudo_sent_datasets import PseudoSents_Dataset
from models.base_model import ViSynoSenseEmbedding
from utils.load_config import load_config
from utils.optimizer import create_optimizer
from utils.loss_fn import InfonceDistillLoss
from trainings.utils import train_model

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

import csv

def load_gloss_dict_from_csv(csv_path):
    gloss_dict = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            synset_id = int(row['synset_id'])
            gloss = row['gloss'].strip()
            if synset_id not in gloss_dict:
                gloss_dict[synset_id] = gloss
    return gloss_dict


def setup_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("--load_ckpts", type=int, default=0, help="Model type")
    parser.add_argument("--only_multiple_el", type=int, default=0, help="Model type")
    args = parser.parse_args()
    return args 
        
if __name__=="__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    args = setup_args()
    print(f"Load From Checkpoint: {bool(args.load_ckpts)}")
    print(f"only_multiple_el: {bool(args.only_multiple_el)}")
    print(f"Device: {device}")

    config = load_config("configs/base.yml")
    
    
    with open(config["data"]["train_path"], "r",encoding="utf-8") as f:
        train_sample = json.load(f)
    with open(config["data"]["valid_path"], "r",encoding="utf-8") as f:
        valid_sample = json.load(f)
        
    gloss_dict = load_gloss_dict_from_csv(config["data"]["gloss_path"])
    
    gloss_enc = SentenceTransformer('dangvantuan/vietnamese-embedding'
                                    ,cache_folder="embeddings/vietnamese_embedding")
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config["base_model"])
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    
    train_set = PseudoSents_Dataset(gloss_enc
                                    ,train_sample, tokenizer,gloss_dict
                                    , is_training=True, 
                                    num_synsets_per_batch=128,samples_per_synset=6,
                                    only_multiple_el=bool(args.only_multiple_el))
    
    valid_set = PseudoSents_Dataset(gloss_enc,valid_sample, tokenizer,gloss_dict, is_training=False
                                    ,only_multiple_el=bool(args.only_multiple_el))
    
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

    if bool(args.load_ckpts):
        model = ViSynoSenseEmbedding.from_pretrained(config["base_model"]).to(device)
    else:
        model = ViSynoSenseEmbedding(
            tokenizer,
            model_name=config["base_model"],
            cache_dir=config["base_model_cache_dir"],
            hidden_dim=config["model"]["hidden_dim"],
            out_dim=config["model"]["out_dim"],
            dropout=config["model"]["dropout"],
            num_layers=config["model"]["num_layers"],
            context_window_size=config["model"]["context_window_size"],
            use_proj=config["model"]["use_proj"],
            polym = config["model"]["use_proj"],
            encoder_type = config["model"]["encoder_type"],
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

    
    loss_fn = InfonceDistillLoss()

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
