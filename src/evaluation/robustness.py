import torch
import numpy as np
import captum
from captum.attr import LayerAttribution

def average_sensitivity(relevance_map, model, input_tensor, label, num_steps=20, 
                       perturbation_mode='noise', baseline='zero', attr_model=None, method='lrp', feature_masks=None):
    """
    Evaluates the stability of explanations by measuring how much they change
    when the input is slightly perturbed.
    
    Lower values indicate more stable (better) explanations.
    
    Processes all samples in a batch simultaneously for better efficiency.
    """
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    # Get original explanation
    original_relevance = relevance_map.clone()
    
    # Calculate norm for normalization
    original_norm = torch.linalg.norm(original_relevance.view(original_relevance.size(0), -1), dim=1)

    
    # Track accumulated sensitivities for each sample
    accumulated_sensitivities = torch.zeros(input_tensor.shape[0], device=input_tensor.device)
    
    # Generate multiple perturbations
    for _ in range(num_steps):
        # Create perturbed input for all samples at once
        if perturbation_mode == 'noise':
            noise = torch.randn_like(input_tensor) * 0.1
            perturbed_input = input_tensor + noise
        else:
            # Other perturbation methods...
            continue
            
        # Get explanation for perturbed input
        if method == 'Lime':
            # LIME might need to be handled separately as it's sample-based
            batch_perturbed_relevance = []
            for b in range(input_tensor.shape[0]):
                perturbed_relevance = attr_model.attribute(perturbed_input[b:b+1], label[b],
                    feature_mask=feature_masks[b:b+1].unsqueeze(1) if feature_masks is not None else None,
                    n_samples=40,
                    perturbations_per_eval=16,
                    show_progress=False)
                batch_perturbed_relevance.append(perturbed_relevance)
            perturbed_relevance = torch.cat(batch_perturbed_relevance, dim=0)
        elif method == 'LayerGradCam':
            # Some attribution methods can handle batches directly
            perturbed_relevance = attr_model.attribute(perturbed_input, target=label)
            perturbed_relevance = LayerAttribution.interpolate(
                    perturbed_relevance, 
                    input_tensor.shape[2:],
                    interpolate_mode='bicubic'  # Use bicubic interpolation instead of default
                )
        elif method == 'IntegratedGradients':
            perturbed_relevance = attr_model.attribute(perturbed_input, target=label,
                baselines=0,
                method='gausslegendre',
                return_convergence_delta=False)
        elif method == 'LRP':
            perturbed_relevance = attr_model.attribute(perturbed_input, target=label)
        else:
            # Assuming compute_relevance can handle batches
            perturbed_relevance = attr_model.compute_relevance(perturbed_input, label)
        
        # Calculate difference between explanations for all samples at once
        diff = torch.norm((original_relevance - perturbed_relevance).view(original_relevance.size(0), -1), dim=1) / original_norm
        if torch.isnan(diff).any():
            print("Warning: NaN values detected in difference calculation")
            # Replace NaN values with zeros
            diff = torch.nan_to_num(diff, nan=0.0)
        
        # Accumulate differences for each sample
        accumulated_sensitivities += diff
    
    # Calculate average sensitivity across all perturbation steps
    avg_sensitivities = accumulated_sensitivities / num_steps
    
    return avg_sensitivities, None  # Return sensitivities and placeholder for second return value

# def average_sensitivity(relevance_map, model, input_tensor, label, num_steps=20, 
#                        perturbation_mode='mask', baseline='zero'):
#     """
#     Evaluates the faithfulness of LRP explanations using average sensitivity.
    
#     Args:
#         relevance_map: The LRP heatmap (B, C, H, W)
#         model: The classification model
#         input_tensor: The original input image (B, C, H, W)
#         label: The true label or batch of labels (int or tensor of shape (B,))
#         num_steps: Number of perturbation steps
#         perturbation_mode: How to perturb the inputs ('mask' or 'noise')
#         baseline: Baseline for masking ('zero' or 'mean')
        
#     Returns:
#         auc: Area under the sensitivity curve (scalar or batch)
#         sensitivities: List of sensitivity values at each step (for each sample in batch)
#     """
#     # Ensure input is batched
#     if len(input_tensor.shape) == 3:
#         input_tensor = input_tensor.unsqueeze(0)
#     if len(relevance_map.shape) == 3:
#         relevance_map = relevance_map.unsqueeze(0)
    
#     # Convert label to tensor if it's not already
#     if isinstance(label, int):
#         label = torch.tensor([label], device=input_tensor.device)
#     elif isinstance(label, torch.Tensor) and label.dim() == 0:
#         label = label.unsqueeze(0)
    
#     batch_size = input_tensor.shape[0]
    
#     # Get original prediction
#     original_output = model(input_tensor).detach()
#     original_probs = torch.softmax(original_output, dim=1)
    
#     # Get probability for each sample's label
#     batch_indices = torch.arange(batch_size, device=input_tensor.device)
#     original_prob = original_probs[batch_indices, label].detach()
    
#     # Determine baseline value for masking
#     if baseline == 'mean':
#         baseline_value = input_tensor.mean(dim=[1, 2, 3], keepdim=True) 
#     else:  # zero
#         baseline_value = 0.0
    
#     # Calculate perturbation sizes
#     fractions = torch.linspace(0, 1, num_steps + 1)[1:]  # Skip 0
    
#     # Track sensitivity for each sample
#     all_sensitivities = []
#     pixels_perturbed = []
    
#     for step, fraction in enumerate(fractions):
#         # Create perturbed input for the whole batch
#         perturbed_input = input_tensor.clone()
        
#         # Process each sample in the batch
#         for b in range(batch_size):
#             # Calculate total pixels for this sample
#             total_pixels = relevance_map[b].numel()
            
#             # Number of pixels to perturb
#             num_pixels = int(fraction * total_pixels)
            
#             if step == 0:  # Only add once per sample
#                 pixels_perturbed.append([])
            
#             pixels_perturbed[b].append(num_pixels / total_pixels)
            
#             # Sort relevance map by magnitude (get indices)
#             flat_relevance = relevance_map[b].flatten()
#             sorted_indices = torch.argsort(torch.abs(flat_relevance), descending=True)
            
#             # Get indices to perturb
#             indices_to_perturb = sorted_indices[:num_pixels]
            
#             # Apply perturbation
#             if perturbation_mode == 'mask':
#                 # Mask with baseline value
#                 if isinstance(baseline_value, torch.Tensor):
#                     # Reshape baseline to match the flattened indices
#                     b_value = baseline_value[b].view(-1).expand_as(perturbed_input[b].view(-1))
#                     perturbed_input[b].view(-1)[indices_to_perturb] = b_value[indices_to_perturb]
#                 else:
#                     perturbed_input[b].view(-1)[indices_to_perturb] = baseline_value
#             elif perturbation_mode == 'noise':
#                 # Add random noise
#                 noise = torch.randn_like(perturbed_input[b].view(-1)[indices_to_perturb]) * 0.1
#                 perturbed_input[b].view(-1)[indices_to_perturb] += noise
        
#         # Get new prediction for the whole batch
#         with torch.no_grad():
#             new_output = model(perturbed_input)
#             new_probs = torch.softmax(new_output, dim=1)
#             new_prob = new_probs[batch_indices, label]
        
#         # Calculate sensitivity as probability change for each sample
#         sensitivity = original_prob - new_prob
        
#         if step == 0:
#             # Initialize lists for each sample
#             all_sensitivities = [[] for _ in range(batch_size)]
        
#         # Store sensitivity for each sample
#         for b in range(batch_size):
#             all_sensitivities[b].append(sensitivity[b].item())
    
#     # Calculate area under the sensitivity curve for each sample
#     aucs = []
#     for b in range(batch_size):
#         sensitivities_np = np.array(all_sensitivities[b])
#         pixels_perturbed_np = np.array(pixels_perturbed[b])
#         auc = np.trapz(sensitivities_np, pixels_perturbed_np)
#         aucs.append(auc)
#     return aucs, all_sensitivities