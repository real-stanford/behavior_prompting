import torch
from torch import nn

def validate_optimizer_parameters(optimizer: torch.optim.Optimizer, model: nn.Module):
    """Validates that the optimizer accounts for all parameters in the model."""
    # Get all parameters in the optimizer
    optimizer_params = set()
    for group in optimizer.param_groups:
        for param in group['params']:
            optimizer_params.add(param)
    
    # Get all parameters in the model
    model_params = set()
    model_param_names = []
    for name, param in model.named_parameters():
        model_params.add(param)
        model_param_names.append(name)
    
    # Find missing parameters
    missing_params = model_params - optimizer_params
    missing_param_names = []
    for name, param in model.named_parameters():
        if param in missing_params:
            missing_param_names.append(name)
    
    optimizer_param_count = len(optimizer_params)
    model_param_count = len(model_params)
    
    if optimizer_param_count != model_param_count:
        if all(name.endswith('_dummy_variable') for name in missing_param_names):
            # it's ok if the dummy variable is not in the optimizer
            return

        print(f"Parameter count mismatch: optimizer has {optimizer_param_count} parameters, "
              f"but model has {model_param_count} parameters.")
        print(f"Missing parameters in optimizer:")
        for name in missing_param_names:
            print(f"  - {name}")
        raise ValueError(
            f"Parameter count mismatch: optimizer has {optimizer_param_count} parameters, "
            f"but model has {model_param_count} parameters. "
            f"This indicates some model parameters are not being optimized."
        )
