import torch
import numpy as np

def average_sensitivity(relevance_map, model, input_tensor, label, num_steps=20, 
                       perturbation_mode='mask', baseline='zero'):
    """
    Evaluates the faithfulness of LRP explanations using average sensitivity.
    
    Args:
        relevance_map: The LRP heatmap (B, C, H, W)
        model: The classification model
        input_tensor: The original input image (B, C, H, W)
        label: The true label or batch of labels (int or tensor of shape (B,))
        num_steps: Number of perturbation steps
        perturbation_mode: How to perturb the inputs ('mask' or 'noise')
        baseline: Baseline for masking ('zero' or 'mean')
        
    Returns:
        auc: Area under the sensitivity curve (scalar or batch)
        sensitivities: List of sensitivity values at each step (for each sample in batch)
    """
    # Ensure input is batched
    if len(input_tensor.shape) == 3:
        input_tensor = input_tensor.unsqueeze(0)
    if len(relevance_map.shape) == 3:
        relevance_map = relevance_map.unsqueeze(0)
    
    # Convert label to tensor if it's not already
    if isinstance(label, int):
        label = torch.tensor([label], device=input_tensor.device)
    elif isinstance(label, torch.Tensor) and label.dim() == 0:
        label = label.unsqueeze(0)
    
    batch_size = input_tensor.shape[0]
    
    # Get original prediction
    original_output = model(input_tensor).detach()
    original_probs = torch.softmax(original_output, dim=1)
    
    # Get probability for each sample's label
    batch_indices = torch.arange(batch_size, device=input_tensor.device)
    original_prob = original_probs[batch_indices, label].detach()
    
    # Determine baseline value for masking
    if baseline == 'mean':
        baseline_value = input_tensor.mean(dim=[1, 2, 3], keepdim=True) 
    else:  # zero
        baseline_value = 0.0
    
    # Calculate perturbation sizes
    fractions = torch.linspace(0, 1, num_steps + 1)[1:]  # Skip 0
    
    # Track sensitivity for each sample
    all_sensitivities = []
    pixels_perturbed = []
    
    for step, fraction in enumerate(fractions):
        # Create perturbed input for the whole batch
        perturbed_input = input_tensor.clone()
        
        # Process each sample in the batch
        for b in range(batch_size):
            # Calculate total pixels for this sample
            total_pixels = relevance_map[b].numel()
            
            # Number of pixels to perturb
            num_pixels = int(fraction * total_pixels)
            
            if step == 0:  # Only add once per sample
                pixels_perturbed.append([])
            
            pixels_perturbed[b].append(num_pixels / total_pixels)
            
            # Sort relevance map by magnitude (get indices)
            flat_relevance = relevance_map[b].flatten()
            sorted_indices = torch.argsort(torch.abs(flat_relevance), descending=True)
            
            # Get indices to perturb
            indices_to_perturb = sorted_indices[:num_pixels]
            
            # Apply perturbation
            if perturbation_mode == 'mask':
                # Mask with baseline value
                if isinstance(baseline_value, torch.Tensor):
                    # Reshape baseline to match the flattened indices
                    b_value = baseline_value[b].view(-1).expand_as(perturbed_input[b].view(-1))
                    perturbed_input[b].view(-1)[indices_to_perturb] = b_value[indices_to_perturb]
                else:
                    perturbed_input[b].view(-1)[indices_to_perturb] = baseline_value
            elif perturbation_mode == 'noise':
                # Add random noise
                noise = torch.randn_like(perturbed_input[b].view(-1)[indices_to_perturb]) * 0.1
                perturbed_input[b].view(-1)[indices_to_perturb] += noise
        
        # Get new prediction for the whole batch
        with torch.no_grad():
            new_output = model(perturbed_input)
            new_probs = torch.softmax(new_output, dim=1)
            new_prob = new_probs[batch_indices, label]
        
        # Calculate sensitivity as probability change for each sample
        sensitivity = original_prob - new_prob
        
        if step == 0:
            # Initialize lists for each sample
            all_sensitivities = [[] for _ in range(batch_size)]
        
        # Store sensitivity for each sample
        for b in range(batch_size):
            all_sensitivities[b].append(sensitivity[b].item())
    
    # Calculate area under the sensitivity curve for each sample
    aucs = []
    for b in range(batch_size):
        sensitivities_np = np.array(all_sensitivities[b])
        pixels_perturbed_np = np.array(pixels_perturbed[b])
        auc = np.trapz(sensitivities_np, pixels_perturbed_np)
        aucs.append(auc)
    return aucs, all_sensitivities