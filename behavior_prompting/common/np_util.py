import numpy as np

def add_batch_dim(data):
    """
    Recursively adds a batch dimension to arrays in a (possibly nested) dictionary.
    
    Args:
        data (dict or array): Dictionary or array to process.
        
    Returns:
        dict or array: The input structure with batch dimensions added to all arrays.
    """
    if isinstance(data, dict):
        # Recursively process dictionaries
        return {key: add_batch_dim(value) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        # Add batch dimension to arrays
        return np.expand_dims(data, axis=0)
    else:
        # Return the item as is for non-arrays
        return data
    
def remove_batch_dim(data, index_to_keep=0):
    """
    Recursively removes a batch dimension from arrays in a (possibly nested) dictionary.
    
    Args:
        data (dict or array): Dictionary or array to process.
        
    Returns:
        dict or array: The input structure with batch entry at index_to_keep kept and all other batch entries removed from all arrays.
    """
    if isinstance(data, dict):
        # Recursively process dictionaries
        return {key: remove_batch_dim(value, index_to_keep) for key, value in data.items()}
    elif isinstance(data, np.ndarray):
        # Remove batch dimension to arrays
        return data[index_to_keep]
    else:
        # Return the item as is for non-arrays
        return data
