from torch.optim import AdamW

def create_optimizer(model, config):
    """
    Create optimizer with different learning rates for base model and custom layers
    
    Args:
        model: The neural network model
        config: Configuration dictionary
        
    Returns:
        AdamW optimizer with parameter groups
    """
    # Parameters that should not have weight decay (biases and layer norms)
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    
    # Separate parameters into base model and custom layers
    base_model_decay = []
    base_model_no_decay = []
    custom_decay = []
    custom_no_decay = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter belongs to base model (PhoBERT)
        if "base_model" in name:
            if any(nd in name for nd in no_decay):
                base_model_no_decay.append(param)
            else:
                base_model_decay.append(param)
        else:
            # Custom layers (FusionBlock, Pooling layers, etc.)
            if any(nd in name for nd in no_decay):
                custom_no_decay.append(param)
            else:
                custom_decay.append(param)
    lr_base = float(config["training"]["optimizer"]["lr_base"])
    lr_custom = float(config["training"]["optimizer"]["lr_custom"])
    weight_decay_val = float(config["training"]["optimizer"]["weight_decay"])

    # Create parameter groups with different learning rates and weight decay
    optimizer_groups = [
        {
            "params": base_model_decay,
            "lr": lr_base,
            "weight_decay": weight_decay_val
        },
        {
            "params": base_model_no_decay,
            "lr": lr_base,
            "weight_decay": 0.0
        },
        {
            "params": custom_decay,
            "lr": lr_custom,
            "weight_decay": weight_decay_val
        },
        {
            "params": custom_no_decay,
            "lr": lr_custom,
            "weight_decay": 0.0
        }
    ]

    
    # Filter out empty parameter groups
    optimizer_groups = [group for group in optimizer_groups if len(group["params"]) > 0]
    
    # Create optimizer
    optimizer = AdamW(
        optimizer_groups,
        eps=float(config["training"]["optimizer"]["eps"])  ,
        betas=config["training"]["optimizer"]["betas"]
    )
    
    # Print parameter group information for debugging
    print("Optimizer parameter groups:")
    for i, group in enumerate(optimizer.param_groups):
        param_count = sum(p.numel() for p in group['params'])
        print(f"  Group {i}: {param_count:,} parameters, LR={float(group['lr']):.2e}, WD={float(group['weight_decay'])}")
    
    return optimizer

def optimizer_for_supersense_pred(model,config):
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    
    base_model_params = []
    embedding_custom_params = []
    prediction_head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if "embedding_model.base_model" in name:
            base_model_params.append((name, param))
        elif "embedding_model" in name:
            embedding_custom_params.append((name, param))
        else:
            prediction_head_params.append((name, param))

    optimizer_groups = []

    if base_model_params:
        decay_params = [p for (n, p) in base_model_params if not any(nd in n for nd in no_decay)]
        no_decay_params = [p for (n, p) in base_model_params if any(nd in n for nd in no_decay)]
        optimizer_groups.append({
            "params": decay_params,
            "lr": float(config["training"]["optimizer"]["lr_base"]),
            "weight_decay": float(config["training"]["optimizer"]["weight_decay"])
        })
        optimizer_groups.append({
            "params": no_decay_params,
            "lr": float(config["training"]["optimizer"]["lr_base"]),
            "weight_decay": 0.0
        })

    if embedding_custom_params:
        decay_params = [p for (n, p) in embedding_custom_params if not any(nd in n for nd in no_decay)]
        no_decay_params = [p for (n, p) in embedding_custom_params if any(nd in n for nd in no_decay)]
        optimizer_groups.append({
            "params": decay_params,
            "lr": float(config["training"]["optimizer"]["lr_custom"]),
            "weight_decay": float(config["training"]["optimizer"]["weight_decay"])
        })
        optimizer_groups.append({
            "params": no_decay_params,
            "lr": float(config["training"]["optimizer"]["lr_custom"]),
            "weight_decay": 0.0
        })

    if prediction_head_params:
        head_lr = float(config["training"]["optimizer"].get("lr_head", 3e-4))  
        
        decay_params = [p for (n, p) in prediction_head_params if not any(nd in n for nd in no_decay)]
        no_decay_params = [p for (n, p) in prediction_head_params if any(nd in n for nd in no_decay)]
        
        optimizer_groups.append({
            "params": decay_params,
            "lr": head_lr,
            "weight_decay": float(config["training"]["optimizer"]["weight_decay"])
        })
        optimizer_groups.append({
            "params": no_decay_params,
            "lr": head_lr,
            "weight_decay": 0.0
        })
        
        optimizer_groups = [group for group in optimizer_groups if len(group["params"]) > 0]
    
        # create optimizer
        optimizer = AdamW(
            optimizer_groups,
            eps=float(config["training"]["optimizer"]["eps"]),
            betas=config["training"]["optimizer"]["betas"]
        )

        print("=" * 50)
        print("OPTIMIZER CONFIGURATION")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        for i, group in enumerate(optimizer_groups):
            params = group["params"]
            count = sum(p.numel() for p in params)
            print(f"Group {i}: {count:,} params | LR = {group['lr']:.0e} | WD = {group['weight_decay']}")
        print("=" * 50)
        
        return optimizer
