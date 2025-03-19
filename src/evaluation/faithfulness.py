import torch
import numpy as np
from evaluation.evaluation_utils import difference, batch_uniform_noise
import time

def faithfulness_correlation(relevance_map, model, input_tensor, label, num_steps=5, **kwargs):
    subset_size = kwargs.get('subset_size', 10)
    # Prepare shapes. Expand relevance_map if not the same shape
    if input_tensor.shape != relevance_map.shape:
        relevance_map = relevance_map.expand_as(input_tensor)

    # Flatten the attributions
    batch_size = input_tensor.shape[0]
    relevance_map_flat = relevance_map.reshape(batch_size, -1)
    n_features = relevance_map_flat.shape[-1]

    # Convert label to tensor if it's a scalar
    if isinstance(label, int) or isinstance(label, float):
        label = torch.tensor([label] * batch_size, device=input_tensor.device)
    elif isinstance(label, torch.Tensor) and label.numel() == 1:
        label = label.expand(batch_size)

    # Predict on input
    y_pred = model(input_tensor)
    # Get probabilities for the target labels for each batch item
    batch_indices = torch.arange(batch_size, device=input_tensor.device)
    y_pred = torch.softmax(y_pred, dim=1)[batch_indices, label]

    pred_deltas = []
    att_sums = []

    input_tensor_shape = input_tensor.shape
    # For each test data point, execute a couple of runs.
    for _ in range(num_steps):
        # Randomly mask by subset size - using torch instead of numpy
        rand_indices = []
        for i in range(batch_size):
            # Get flattened indices for this specific batch item
            flat_indices = torch.randperm(n_features)[:subset_size]
            # Convert to indices that are valid for the specific dimensions of the tensor
            rand_indices.append(flat_indices)
        a_ix = torch.stack(rand_indices, dim=0)
        
        x_perturbed = batch_uniform_noise(
            arr=input_tensor.reshape(batch_size, -1),
            indices=a_ix,
        )
            
        x_perturbed = x_perturbed.reshape(*input_tensor_shape)

        # check if the perturbed input is the same as the original input
        if torch.allclose(x_perturbed, input_tensor):
            print("Perturbed input is the same as the original input")

        # Predict on perturbed input
        y_pred_perturb = model(x_perturbed)
        y_pred_perturb = torch.softmax(y_pred_perturb, dim=1)[batch_indices, label]
        
        pred_deltas.append(y_pred - y_pred_perturb)

        # Sum attributions of the random subset
        batch_att_sums = []
        for i in range(batch_size):
            batch_att_sums.append(relevance_map_flat[i, a_ix[i]].sum())
        att_sums.append(torch.tensor(batch_att_sums, device=relevance_map.device))

    pred_deltas = torch.stack(pred_deltas, dim=1)  # Shape: [batch_size, num_steps]
    att_sums = torch.stack(att_sums, dim=1)  # Shape: [batch_size, num_steps]

    # Calculate difference between attribution sums and prediction deltas
    similarity = torch.abs(difference(a=att_sums, b=pred_deltas, batched=True))
    
    # Take the mean across the steps dimension to get one value per batch item
    similarity_mean = torch.mean(similarity, dim=1)

    return similarity_mean

def pixel_flipping(relevance_map, model, input_tensor, label, num_steps=5):
    """
    Evaluates the faithfulness of LRP explanations using pixel flipping.

    Args:
        relevance_map: The LRP heatmap (B, C, H, W)
        model: The classification model
        input_tensor: The original input image (B, C, H, W)
        label: The true label or labels (int or tensor of shape (B,))
        num_steps: Number of occlusion steps
        
    Returns:
        Scores across all batch items
    """
    # Ensure input is batched
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    if len(relevance_map.shape) == 3:
        relevance_map = relevance_map.unsqueeze(0)
    
    batch_size = input_tensor.shape[0]
    
    # Convert label to tensor if it's a scalar
    if isinstance(label, int) or isinstance(label, float):
        label = torch.tensor([label] * batch_size, device=input_tensor.device)
    elif isinstance(label, torch.Tensor) and label.numel() == 1:
        label = label.expand(batch_size)
    
    # Get original predictions
    original_output = model(input_tensor).detach()
    original_probs = torch.softmax(original_output, dim=1)
    
    # Get probabilities for the target labels for each batch item
    batch_indices = torch.arange(batch_size, device=input_tensor.device)
    original_probs = original_probs[batch_indices, label]
    
    # Calculate number of pixels to occlude at each step for each batch item
    total_pixels = relevance_map[0].numel()  # Pixels per batch item
    step_size = total_pixels // num_steps
    
    # Store scores for each batch item
    batch_scores = []
    input_clone = input_tensor.clone()
    
    for i in range(num_steps):
        step_scores = []
        
        for b in range(batch_size):
            # Get top indices from relevance map for this batch item
            top_indices = relevance_map[b].view(-1).topk(step_size, largest=True).indices
            
            # Occlude pixels in input for this batch item
            input_clone[b].view(-1)[top_indices] = 0
            
        # Get new predictions for all batch items at once
        new_output = model(input_clone).detach()
        new_probs = torch.softmax(new_output, dim=1)
        new_probs = new_probs[batch_indices, label]
        
        # Calculate scores as probability drop for each batch item
        scores = original_probs - new_probs
        batch_scores.append(scores)
    
    # Stack scores across steps and calculate mean for each batch item
    batch_scores = torch.stack(batch_scores)  # Shape: [num_steps, batch_size]
    mean_scores = torch.mean(batch_scores, dim=1)
    
    return mean_scores
