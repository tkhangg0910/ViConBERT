import json
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.utils import is_torch_available
from transformers import PreTrainedTokenizerFast, PhobertTokenizerFast
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data.processed.stage2_supersense_pred.supersense_pred_dataset import SuperSenseDataset
from models.stage_2_model import SuperSensePredModel
from models.base_model import SynoViSenseEmbeddingV2

from utils.load_config import load_config
from utils.optimizer import optimizer_for_supersense_pred
from utils.loss_fn import InfoNceLoss
from trainings.stage_2.utils import train_model

if is_torch_available() and torch.multiprocessing.get_start_method() == "fork":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# def setup_args():
#     parser = argparse.ArgumentParser(description="Train a model")
#     # parser.add_argument("--model", type=str, default="v2", help="Model type")
#     parser.add_argument("--load_ckpts", type=int, default=0, help="Model type")
#     args = parser.parse_args()
#     return args 
        
if __name__=="__main__":
    # args = setup_args()
    # print(bool(args.load_ckpts))

    config = load_config("configs/stage2.yml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    with open(config["data"]["train_path"], "r",encoding="utf-8") as f:
        train_sample = json.load(f)
    with open(config["data"]["valid_path"], "r",encoding="utf-8") as f:
        valid_sample = json.load(f)
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config["base_model"])
    
        
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    
    
    train_set = SuperSenseDataset(train_sample, tokenizer)
    valid_set = SuperSenseDataset(valid_sample, tokenizer)
    
    train_dataloader = DataLoader(train_set,
                                  config["training"]["batch_size"],
                                  shuffle=True,
                                  collate_fn=train_set.collate_fn,
                                  num_workers=config["data"]["num_workers"],
                                  pin_memory=True
                                  )
    valid_dataloader = DataLoader(valid_set,
                                  config["training"]["batch_size"],
                                  shuffle=False,
                                  collate_fn=valid_set.collate_fn,
                                  num_workers=config["data"]["num_workers"],
                                  pin_memory=True
                                  )
    embedding_model = SynoViSenseEmbeddingV2.from_pretrained(config["base_model"])
    
    model = SuperSensePredModel(embedding_model,
                                supersense_size=config["model"]["supersense_size"],
                                pred_head_num_layer=config["model"]["pred_head_num_layer"],
                                prediction_hidden_dim=config["model"]["prediction_hidden_dim"]
                                )
    
    total_steps = len(train_dataloader) * config["training"]["epochs"] 
    steps_per_epoch = len(train_dataloader)
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps}")
    warmup_steps = int(0.1 * total_steps)
    
    optim = optimizer_for_supersense_pred(model, config)
    assert optim is not None, "Optimizer must not be None"

    scheduler = ReduceLROnPlateau(
        optimizer = optim,
        mode='min',         
        factor=0.5,         
        patience=2,         
        min_lr=1e-6          
    )

    
    loss_fn = nn.CrossEntropyLoss()

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
