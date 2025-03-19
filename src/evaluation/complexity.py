import torch
import numpy as np

def sparseness(relevance_map):
    """
    Evaluates the sparseness of LRP explanations using the Hoyer metric.
    
    Hoyer(x) = (sqrt(n) - (sum(|x|) / sqrt(sum(x^2)))) / (sqrt(n) - 1)
    
    Args:
        relevance_map: The LRP heatmap (B, C, H, W)
        
    Returns:
        sparseness_score: Value between 0 and 1, where 1 is maximally sparse
    """
    # Ensure input is batched and flatten relevance map
    if len(relevance_map.shape) == 3:
        relevance_map = relevance_map.unsqueeze(0)
    
    # Calculate sparseness for each example in the batch
    batch_size = relevance_map.shape[0]
    sparseness_scores = []
    
    for i in range(batch_size):
        # Flatten the relevance map
        rel_flat = relevance_map[i].flatten()
        n = rel_flat.numel()
        
        # Calculate L1 norm
        l1_norm = torch.sum(torch.abs(rel_flat))
        
        # Calculate L2 norm
        l2_norm = torch.sqrt(torch.sum(rel_flat ** 2))
        
        # Calculate Hoyer sparseness
        if l2_norm > 0:  # Prevent division by zero
            hoyer = (np.sqrt(n) - (l1_norm / l2_norm)) / (np.sqrt(n) - 1)
        else:
            hoyer = 0
            
        sparseness_scores.append(abs(hoyer.item()))
    return sparseness_scores
    # Return mean sparseness if batch, or single value if one example
    # if batch_size == 1:
    #     return sparseness_scores[0]
    # else:
    #     return np.mean(sparseness_scores)
