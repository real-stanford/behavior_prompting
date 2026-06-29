from typing import Dict, Callable, Union, Any, List
import torch
import collections

def add_batch_dim(data):
    """
    Recursively adds a batch dimension to tensors in a (possibly nested) dictionary.
    
    Args:
        data (dict or tensor): Dictionary or tensor to process.
        
    Returns:
        dict or tensor: The input structure with batch dimensions added to all tensors.
    """
    if isinstance(data, dict):
        # Recursively process dictionaries
        return {key: add_batch_dim(value) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        # Add batch dimension to tensors
        return data.unsqueeze(0)
    else:
        # Return the item as is for non-tensors
        return data

def move_batch_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_batch_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_batch_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_batch_to_device(item, device) for item in data)
    else:
        return data  # leave non-tensors unchanged
    
def remove_batch_dim(data, index_to_keep=0):
    """
    Recursively removes a batch dimension from tensors in a (possibly nested) dictionary.
    
    Args:
        data (dict or tensor): Dictionary or tensor to process.
        
    Returns:
        dict or tensor: The input structure with batch entry at index_to_keep kept and all other batch entries removed from all tensors.
    """
    if isinstance(data, dict):
        # Recursively process dictionaries
        return {key: remove_batch_dim(value, index_to_keep) for key, value in data.items()}
    elif isinstance(data, torch.Tensor):
        # Remove batch dimension to tensors
        return data[index_to_keep]
    else:
        # Return the item as is for non-tensors
        return data

def dict_apply(
        x: Dict[str, torch.Tensor], 
        func: Callable[[torch.Tensor], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result
    
def remove_batch_dim_from_prompt(prompt, index_to_keep=0):
    """
    Prompts need a special method to remove batch dimension because they are padded. This functin trims the padding out of the prompt after removing the batch dimension.
    """
    mask = prompt['metadata']['mask'][index_to_keep]
    invalid_locations = torch.where(mask)[0]
    if len(invalid_locations) > 0:
        invalid_location = invalid_locations[0]
    else:
        invalid_location = len(mask)
    
    # select just this batch entry
    prompt = remove_batch_dim(prompt, index_to_keep)

    # trim all the entries to the invalid location
    prompt['obs'] = dict_apply(prompt['obs'], lambda x: x[:invalid_location])
    prompt['action'] = prompt['action'][:invalid_location]

    return prompt
    
def move_batch_to_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: move_batch_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_batch_to_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_batch_to_numpy(item) for item in data)
    else:
        return data

def dict_apply_split(
        x: Dict[str, torch.Tensor], 
        split_func: Callable[[torch.Tensor], Dict[str, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
    results = collections.defaultdict(dict)
    for key, value in x.items():
        result = split_func(value)
        for k, v in result.items():
            results[k][key] = v
    return results

def dict_apply_reduce(
        x: List[Dict[str, torch.Tensor]],
        reduce_func: Callable[[List[torch.Tensor]], torch.Tensor]
        ) -> Dict[str, torch.Tensor]:
    result = dict()
    for key in x[0].keys():
        result[key] = reduce_func([x_[key] for x_ in x])
    return result
