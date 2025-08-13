from torch.optim import AdamW

def create__diff_optimizer(model, config):
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
        if "context_encoder" in name:
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

def create_optimizer(model, config):
    """
    Create AdamW optimizer with a single learning rate for all parameters.
    
    Args:
        model: The neural network model
        config: Configuration dictionary
        
    Returns:
        AdamW optimizer
    """
    lr = float(config["training"]["optimizer"]["lr"])
    weight_decay_val = float(config["training"]["optimizer"]["weight_decay"])
    
    # Chỉ chọn những parameter requires_grad
    params = [p for p in model.parameters() if p.requires_grad]
    
    optimizer = AdamW(
        params,
        lr=lr,
        weight_decay=weight_decay_val,
        eps=float(config["training"]["optimizer"].get("eps", 1e-8)),
        betas=config["training"]["optimizer"].get("betas", (0.9, 0.999))
    )
    
    print(f"Optimizer created: {sum(p.numel() for p in params):,} parameters, LR={lr:.2e}, WD={weight_decay_val}")
    
    return optimizer
