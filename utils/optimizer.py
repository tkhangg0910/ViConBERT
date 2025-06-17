from torch.optim import AdamW

def create_optimizer(model, config):
    no_decay = ["bias", "LayerNorm.weight"]
    base_model_params = []
    custom_params = []
    
    for name, param in model.named_parameters():
        if "base_model" in name:
            if any(nd in name for nd in no_decay):
                base_model_params.append(param)
            else:
                base_model_params.append(param)
        else:
            custom_params.append(param)
    
    optimizer_groups = [
        {
            "params": base_model_params,
            "lr": config["training"]["optimizer"]["lr_base"],
            "weight_decay": config["training"]["optimizer"]["weight_decay"]
        },
        {
            "params": custom_params,
            "lr": config["training"]["optimizer"]["lr_custom"],
            "weight_decay": config["training"]["optimizer"]["weight_decay"]
        }
    ]
    
    return AdamW(
        optimizer_groups,
        eps=config["training"]["optimizer"]["eps"],
        betas=config["training"]["optimizer"]["betas"]
    )
