import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from matplotlib.colors import Normalize
import random

from lrp import UniformRuleLRP
from models import load_model
from datasets import get_mnist_data, get_imagenet_data
from utils import plot_heatmap

import quantus
from skimage.segmentation import slic
from captum.attr import LayerAttribution, LayerGradCam, Lime, IntegratedGradients, GuidedGradCam, LRP
from torchvision.models import vgg16, VGG16_Weights

# Define constants for rule types if needed (copied from combined_lrp_visualizer)
AVAILABLE_RULES = ['alpha_beta', 'epsilon', 'gamma']

def compare_xai_methods(
    model,
    images,
    labels,
    output_dir=None,
    batch_size=10,
    loaded_lrp_params=None,
    lrp_weights=None,
    diff_top_n=5,
    overlay=False,
    clamp_percentile=None,
    device=torch.device('cpu')
):
    """
    Compare visualizations using different XAI methods (GradCAM, Lime, IG, CombinedLRP_Difference).

    Args:
        model: The neural network model (on the correct device)
        images: Input images to visualize (B, C, H, W) (on the correct device)
        labels: Labels for the images (on the correct device)
        output_dir: Directory to save visualizations
        batch_size: Number of examples to visualize
        loaded_lrp_params: Dict mapping metric name to loaded parameter data
        lrp_weights: Dict mapping metric name to float weight
        diff_top_n: Number of top other classes to average for difference map.
        overlay: If True, overlay heatmaps on original images
        clamp_percentile: Specify the percentile (e.g., 10 for 10th/90th) to clamp CombinedLRP_Difference heatmap before visualization. No clamping if not set.
        device: The torch device to use for computations
    """
    if loaded_lrp_params is None: loaded_lrp_params = {}
    if lrp_weights is None: lrp_weights = {}

    # Load ImageNet class labels
    try:
        import json
        with open('/home/neuro/XAIProject/data/imagenet_class_index.json') as f:
            class_idx = json.load(f)
        idx_to_labels = {int(k): v[1] for k, v in class_idx.items()}
    except:
        # Fallback if file not found
        idx_to_labels = None
        print("Warning: ImageNet class labels file not found. Using numeric labels only.")
    
    # Limit number of examples
    batch_size = min(batch_size, len(images))
    
    images_batch = images[:batch_size]
    labels_batch = labels[:batch_size]
    
    # Create feature masks for each image in the batch for LIME
    feature_masks = []
    for img in images_batch:
        # Convert single image to numpy format (C, H, W) -> (H, W, C)
        x_numpy = img.cpu().permute(1, 2, 0).numpy()
        # Handle both grayscale and RGB images
        if x_numpy.shape[2] == 1:
            x_numpy = x_numpy.squeeze(2)  # Remove channel dim for grayscale
        feature_mask = slic(x_numpy, n_segments=100, compactness=10, sigma=0.5)
        feature_masks.append(feature_mask)
    feature_masks = torch.tensor(np.stack(feature_masks))
    # Move feature_masks to the correct device
    feature_masks = feature_masks.to(device)
    
    # Initialize XAI methods
    # For GradCAM, find a suitable target layer
    target_layer = None
    if hasattr(model, 'features'):
        # For VGG-like models
        target_layer = model.features[-3]  # Last convolutional layer
    
    # Initialize attributions dictionary
    attributions = {}

    # Compute attributions for each method
    with torch.no_grad():
        # GradCAM
        if target_layer:
            gradcam = LayerGradCam(model, target_layer)
            gradcam_attr = gradcam.attribute(
                images_batch,
                target=labels_batch
            )
            
            # guided
            # guided_gradcam = GuidedGradCam(model, target_layer)
            # gradcam_attr = guided_gradcam.attribute(
            #     images[:batch_size],
            #     target=labels[:batch_size]
            # )
            # Use a more sophisticated interpolation method
            gradcam_attr = LayerAttribution.interpolate(
                gradcam_attr, 
                images_batch.shape[2:],
                interpolate_mode='bicubic'  # Use bicubic interpolation instead of default
            )
            attributions['GradCAM'] = gradcam_attr.detach().cpu().numpy()
        
        # LIME
        lime = Lime(model)
        lime_attr = lime.attribute(
            images_batch,
            target=labels_batch,
            feature_mask=feature_masks.unsqueeze(1),
            n_samples=40,
            perturbations_per_eval=16,
            show_progress=False
        )
        attributions['LIME'] = lime_attr.detach().cpu().numpy()
        
        # IntegratedGradients
        ig = IntegratedGradients(model)
        ig_attr, _ = ig.attribute(
            images_batch,
            baselines=0,
            target=labels_batch,
            method='gausslegendre',
            return_convergence_delta=True
        )
        attributions['IntegratedGradients'] = ig_attr.detach().cpu().numpy()

        lrp_captum = LRP(model)
        lrp_attr = lrp_captum.attribute(images_batch, target=labels_batch)
        attributions['LRP_Captum'] = lrp_attr.detach().cpu().numpy()

        # with open('saved_params/opt_epsilon_faithfulness.json', 'r') as f:
        #     epsilon_params = torch.tensor(json.load(f))
        # lrp = UniformRuleLRP(model, rule_type='epsilon', rule_params=epsilon_params)
        # # lrp = UniformValueLRP(model, rule_types=optimized_params, rule_values={'alpha_beta': 2.0, 'epsilon': 0.25, 'gamma': 0.25, 'zero': 0.0})
        # lrp_attr = lrp.compute_relevance(images_batch, labels_batch)
        # attributions['URO_EPSILON'] = lrp_attr.detach().cpu().numpy()

    # --- Compute Combined LRP Difference Attribution ---
    print(f"\n--- Starting Combined LRP Difference Calculation (Top N = {diff_top_n}) ---")
    all_difference_maps = {} # Store difference maps for each LRP model: {name: diff_map_batch}

    # Get model predictions for the batch ONCE
    try:
        with torch.no_grad():
            outputs = model(images_batch)
        probs, indices = torch.sort(torch.softmax(outputs, dim=1), descending=True)
        print("  Computed model predictions for the batch.")
    except Exception as e:
        print(f"Error: Failed to get model predictions: {e}. Skipping Combined LRP Difference.")
        # Skip the rest of the LRP calculation if predictions fail
        loaded_lrp_params = {} # Effectively disable the loop below

    # Loop through each LRP parameter set that has a non-zero weight
    active_lrp_names = [name for name, params in loaded_lrp_params.items() if lrp_weights.get(name, 0.0) != 0.0]

    for name in active_lrp_names:
        params_data = loaded_lrp_params[name]
        weight = lrp_weights[name] # We know weight is non-zero here
        print(f"\n  Processing LRP model: '{name}' (Weight: {weight:.2f})")

        try:
            # --- Instantiate LRP Model (reuse logic from previous step) ---
            rule_type = None
            parameters_list = None
            if isinstance(params_data, dict):
                rule_type = params_data.get('rule_type')
                parameters_list = params_data.get('parameters')
                if parameters_list is None: raise ValueError("'parameters' key missing")
            elif isinstance(params_data, list):
                parameters_list = params_data
                inferred = False
                for available_rule in AVAILABLE_RULES:
                    if available_rule in name: rule_type = available_rule; inferred = True; break
                if not inferred: rule_type = 'alpha_beta'; print(f"    Warning: Assuming rule '{rule_type}' for list format.")
            else: raise TypeError(f"Unsupported data type {type(params_data)}")

            if rule_type is None: rule_type = 'alpha_beta'; print(f"    Warning: Assuming rule '{rule_type}'.")
            if rule_type not in AVAILABLE_RULES: raise ValueError(f"Unsupported rule type '{rule_type}'")

            parameters = torch.tensor(parameters_list, device=device)
            lrp_instance = UniformRuleLRP(model, rule_type=rule_type, rule_params=parameters, correct_class_only=False)
            print(f"    Instantiated UniformRuleLRP with rule '{rule_type}'.")
            # --- End LRP Instantiation ---

            current_model_diff_maps_batch = [] # Store diff maps for this model across the batch

            # Loop through each image in the batch
            for i in range(batch_size):
                image_single = images_batch[i:i+1] # Keep batch dim
                target_label_single = labels_batch[i:i+1]
                target_idx = target_label_single.item()
                # print(f"      Image {i}, Target Class: {target_idx}")

                # 1. Calculate relevance for the target class
                target_relevance = lrp_instance.compute_relevance(image_single, target_label_single)
                # print(f"        Target relevance computed. Sum: {target_relevance.sum():.4f}")

                # 2. Calculate relevance for top N *other* predicted classes
                top_n_relevance_maps_for_image = []
                collected_count = 0
                for k in range(outputs.shape[1]): # Iterate through all possible class indices
                    pred_idx = indices[i, k].item()

                    if pred_idx != target_idx:
                        pred_label_single = torch.tensor([pred_idx], device=device)
                        pred_relevance = lrp_instance.compute_relevance(image_single, pred_label_single)
                        top_n_relevance_maps_for_image.append(pred_relevance)
                        collected_count += 1
                        # print(f"        Computed relevance for other class {pred_idx}. Count: {collected_count}")
                        if collected_count == diff_top_n:
                            break # Stop once we have enough

                # 3. Average the top N relevance maps
                if top_n_relevance_maps_for_image:
                    # Stack along a new dimension (dim=0) and then average
                    stacked_top_n = torch.stack(top_n_relevance_maps_for_image, dim=0)
                    avg_top_n_relevance = torch.mean(stacked_top_n, dim=0)
                    # print(f"        Averaged relevance of {len(top_n_relevance_maps_for_image)} other classes. Avg Sum: {avg_top_n_relevance.sum():.4f}")
                else:
                    # Handle case where diff_top_n is 0 or no other classes were predicted
                    avg_top_n_relevance = torch.zeros_like(target_relevance)
                    print(f"        Warning: No other class relevance maps found/averaged for image {i}.")

                # 4. Calculate the difference map for this image and this LRP model
                diff_map = target_relevance - avg_top_n_relevance
                current_model_diff_maps_batch.append(diff_map)
                # print(f"        Difference map calculated. Sum: {diff_map.sum():.4f}")

            # Store the batch of difference maps for the current LRP model
            if current_model_diff_maps_batch:
                 # Concatenate along the batch dimension (dim=0)
                 all_difference_maps[name] = torch.cat(current_model_diff_maps_batch, dim=0)
                 print(f"    Finished processing batch for '{name}'. Stored difference maps.")
            else:
                 print(f"    Warning: No difference maps generated for model '{name}'.")


        except Exception as e:
            print(f"Warning: Failed processing LRP model '{name}': {e}. Skipping this model.")
            if name in all_difference_maps: # Clean up partial results if error occurred mid-batch
                del all_difference_maps[name]

    # --- Combine the difference maps across models ---
    final_combined_diff_map = None
    if all_difference_maps:
        print("\n  Combining difference maps across models...")
        first_map_shape = next(iter(all_difference_maps.values())).shape
        final_combined_diff_map_tensor = torch.zeros(first_map_shape, device=device) # Initialize on correct device
        contributing_models = 0
        for name, diff_maps_batch in all_difference_maps.items():
            weight = lrp_weights.get(name, 0.0) # Should be non-zero based on earlier filter
            if diff_maps_batch.shape == first_map_shape:
                final_combined_diff_map_tensor += diff_maps_batch * weight
                print(f"    Added weighted difference map from '{name}' (Weight: {weight:.2f})")
                contributing_models += 1
            else:
                print(f"    Warning: Skipping difference map from '{name}' due to shape mismatch ({diff_maps_batch.shape} vs {first_map_shape}).")

        if contributing_models > 0:
            attributions['CombinedLRP_Difference'] = final_combined_diff_map_tensor.detach().cpu().numpy()
            print(f"  Successfully combined difference maps from {contributing_models} LRP models.")
        else:
            print("  Warning: No difference maps could be combined.")
    else:
        print("Warning: No difference maps were generated by any LRP model. Cannot compute final combined map.")

    # --- Plotting ---
    # The existing plotting loop should work, it just needs the new 'CombinedLRP_Difference' key
    # Update the number of columns calculation based on the final `attributions` dict
    num_methods_to_plot = len(attributions)
    num_cols_single = num_methods_to_plot + 1
    num_cols_summary = num_methods_to_plot + 1

    print(f"\n--- Generating Visualizations ({num_methods_to_plot} methods + Input) ---")

    # --- Create visualizations for each example (Single Plot per Image) ---
    for i in range(batch_size):
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(1, 6, 1)
        # Convert image from (C, H, W) to (H, W, C) format for matplotlib
        img_display = images[i].cpu().permute(1, 2, 0).numpy()
        
        # Handle both grayscale and RGB images
        if img_display.shape[2] == 1:
            plt.imshow(img_display.squeeze(2), cmap='gray')
        else:
            # For ImageNet models, we might need to denormalize
            # Assuming standard ImageNet normalization
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
            img_display = img_display * std + mean
            
            # Clip values to ensure they're in valid range
            img_display = np.clip(img_display, 0, 1)
            
            plt.imshow(img_display)
        
        # Get class name if available
        label_text = f"Label: {labels[i].item()}"
        if idx_to_labels is not None and labels[i].item() in idx_to_labels:
            label_text = f"Label: {idx_to_labels[labels[i].item()]}"
            
        plt.title(f"Input Image\n{label_text}")
        plt.axis('off')
        
        # Plot each XAI method
        for idx, (method_name, attr) in enumerate(attributions.items(), 2):
            plt.subplot(1, 6, idx)
            
            heatmap = attr[i].squeeze()
            # # Handle different attribution formats
            if heatmap.ndim == 3:
                heatmap = np.mean(heatmap, axis=0)

            # --- Apply Percentile Clamping if requested --- 
            plot_title = method_name
            methods_to_clamp = ['CombinedLRP_Difference', 'IntegratedGradients', 'LRP_Captum']
            if method_name in methods_to_clamp and clamp_percentile is not None:
                try:
                    lower_perc = np.percentile(heatmap, clamp_percentile)
                    upper_perc = np.percentile(heatmap, 100 - clamp_percentile)
                    heatmap = np.clip(heatmap, lower_perc, upper_perc)
                    plot_title = f"{method_name}\n(Clamped {clamp_percentile}%) "
                    print(f"  Applied {clamp_percentile}% clamping to {method_name} heatmap for image {i}. New range: [{heatmap.min():.4f}, {heatmap.max():.4f}]")
                except Exception as e:
                    print(f"  Warning: Failed to apply percentile clamping for {method_name} on image {i}: {e}")
            # --- End Clamping ---
            
            # Normalize for visualization (based on potentially clamped data)
            vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
            if vmax < 1e-9: # Avoid division by zero or near-zero
                vmax = 1.0
            vmin = -vmax
            
            if overlay:
                # Create a colormap for the heatmap
                cmap = plt.cm.bwr
                norm = Normalize(vmin=vmin, vmax=vmax)
                heatmap_colored = cmap(norm(heatmap))
                
                # Set alpha channel based on absolute magnitude of relevance
                # Increase the alpha values to make heatmap more visible
                alpha = np.abs(heatmap) / vmax # Use potentially clamped map for alpha
                alpha = np.clip(alpha * 2.0, 0.3, 0.9)  # Increased minimum alpha and scaling factor
                heatmap_colored[..., 3] = alpha
                
                # Display original image
                if img_display.shape[2] == 1:
                    plt.imshow(img_display.squeeze(2), cmap='gray')
                else:
                    plt.imshow(img_display)
                
                # Overlay heatmap with higher overall opacity
                plt.imshow(heatmap_colored, alpha=0.8)  # Increased from 0.7 to 0.8
            else:
                # Original visualization without overlay
                plt.imshow(heatmap, cmap='bwr', vmin=vmin, vmax=vmax) # Plot potentially clamped map
            
            plt.colorbar()
            plt.title(plot_title) # Use potentially modified title
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save or display the figure
        if output_dir:
            output_path = Path(output_dir) / f"xai_comparison_example_{i}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    # Create a summary image with multiple examples in a grid
    if batch_size > 1:
        # Number of methods plus the original image
        num_cols = len(attributions) + 1
        
        plt.figure(figsize=(num_cols * 3, batch_size * 3))
        
        for i in range(batch_size):
            # Original image - place at the start of each row
            plt.subplot(batch_size, num_cols, i * num_cols + 1)
            img_display = images[i].cpu().permute(1, 2, 0).numpy()
            
            # Handle both grayscale and RGB images
            if img_display.shape[2] == 1:
                plt.imshow(img_display.squeeze(2), cmap='gray')
            else:
                # For ImageNet models, we might need to denormalize
                # Assuming standard ImageNet normalization
                mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
                std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
                img_display = img_display * std + mean
                
                # Clip values to ensure they're in valid range
                img_display = np.clip(img_display, 0, 1)
                
                plt.imshow(img_display)
            label_text = f"Label: {labels[i].item()}"
            if idx_to_labels is not None and labels[i].item() in idx_to_labels:
                label_text = f"Label: {idx_to_labels[labels[i].item()]}"
            plt.title(f"{label_text}")
            plt.axis('off')
            
            # Plot each XAI method - place in columns after the original image
            for idx, (method_name, attr) in enumerate(attributions.items()):
                plt.subplot(batch_size, num_cols, i * num_cols + idx + 2)
                
                heatmap = attr[i].squeeze()
                
                # Handle different attribution formats
                if heatmap.ndim == 3:
                    heatmap = np.mean(heatmap, axis=0)
                
                # --- Apply Percentile Clamping if requested --- 
                plot_title = method_name
                heatmap_to_plot = heatmap # Default to original heatmap
                methods_to_clamp = ['CombinedLRP_Difference', 'IntegratedGradients', 'LRP_Captum']
                if method_name in methods_to_clamp and clamp_percentile is not None:
                    try:
                        # Use a copy to avoid modifying the original data used by other plots if needed
                        clamped_heatmap = heatmap.copy()
                        lower_perc = np.percentile(clamped_heatmap, clamp_percentile)
                        upper_perc = np.percentile(clamped_heatmap, 100 - clamp_percentile)
                        clamped_heatmap = np.clip(clamped_heatmap, lower_perc, upper_perc)
                        heatmap_to_plot = clamped_heatmap # Use the clamped version for plotting
                        plot_title = f"{method_name}\n(Clamped {clamp_percentile}%) "
                        # Maybe reduce verbosity for the summary plot
                        # print(f"  Applied {clamp_percentile}% clamping to {method_name} heatmap for image {i} (summary). New range: [{heatmap_to_plot.min():.4f}, {heatmap_to_plot.max():.4f}]")
                    except Exception as e:
                        print(f"  Warning: Failed to apply percentile clamping for {method_name} on image {i} (summary): {e}")
                        # heatmap_to_plot remains the original heatmap in case of error
                # --- End Clamping ---
                
                # Normalize for visualization (based on potentially clamped data in heatmap_to_plot)
                vmax = max(abs(np.max(heatmap_to_plot)), abs(np.min(heatmap_to_plot)))
                if vmax < 1e-9: # Avoid division by zero or near-zero
                    vmax = 1.0
                vmin = -vmax
                
                if overlay:
                    # Create a colormap for the heatmap
                    cmap = plt.cm.bwr
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    heatmap_colored = cmap(norm(heatmap_to_plot))
                    
                    # Set alpha channel based on absolute magnitude of relevance
                    # Increase the alpha values to make heatmap more visible
                    alpha = np.abs(heatmap_to_plot) / vmax # Use potentially clamped map for alpha
                    alpha = np.clip(alpha * 2.0, 0.3, 0.9)  # Increased minimum alpha and scaling factor
                    heatmap_colored[..., 3] = alpha
                    
                    # Display original image
                    if img_display.shape[2] == 1:
                        plt.imshow(img_display.squeeze(2), cmap='gray')
                    else:
                        plt.imshow(img_display)
                    
                    # Overlay heatmap with higher overall opacity
                    plt.imshow(heatmap_colored, alpha=0.8)  # Increased from 0.7 to 0.8
                else:
                    # Original visualization without overlay
                    plt.imshow(heatmap_to_plot, cmap='bwr', vmin=vmin, vmax=vmax) # Plot potentially clamped map
                
                plt.title(plot_title) # Use potentially modified title
                plt.axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "xai_methods_comparison_summary.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare different XAI visualization methods including Combined LRP Difference')
    parser.add_argument('--model_path', type=str, default='models/mnist_cnn.pth', help='Path to the model')
    parser.add_argument('--output_dir', type=str, default='xai_comparisons', help='Output directory for visualizations')

    # --- Add Combined LRP Args ---
    lrp_group = parser.add_argument_group('Combined LRP Parameters & Weights')
    lrp_group.add_argument('--params-sensitivity', type=str, default='saved_params/opt_alpha_beta_average_sensitivity.json',
                           help='Path to LRP parameters JSON file optimized for Sensitivity.')
    lrp_group.add_argument('--params-sparseness', type=str, default='saved_params/opt_alpha_beta_sparseness.json',
                           help='Path to LRP parameters JSON file optimized for Sparseness.')
    lrp_group.add_argument('--params-faithfulness', type=str, default='saved_params/opt_alpha_beta_faithfulness.json',
                           help='Path to LRP parameters JSON file optimized for Faithfulness.')
    lrp_group.add_argument('--weight-sensitivity', type=float, default=1.0,
                           help='Weight for the sensitivity map in the combination.')
    lrp_group.add_argument('--weight-sparseness', type=float, default=1.0,
                           help='Weight for the sparseness map in the combination.')
    lrp_group.add_argument('--weight-faithfulness', type=float, default=2.0,
                           help='Weight for the faithfulness map in the combination.')
    lrp_group.add_argument('--diff_top_n', type=int, default=100,
                           help='Number of top other predicted classes to average for difference calculation.')
    # --- End Combined LRP Args ---

    parser.add_argument('--batch_size', type=int, default=5, help='Number of examples to visualize')
    parser.add_argument('--input_dir', type=str, default='/home/neuro/XAIProject/data/ImageNet', help='Input directory for images')
    parser.add_argument('--overlay', action='store_true', help='Overlay heatmaps on original images')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--clamp-percentile', type=int, default=None,
                        help='Specify the percentile (e.g., 10 for 10th/90th) to clamp CombinedLRP_Difference heatmap before visualization. No clamping if not set.')

    args = parser.parse_args()

    # --- Validate diff_top_n ---
    if args.diff_top_n <= 0:
        parser.error("--diff_top_n must be a positive integer.")

    # --- Validate clamp_percentile ---
    if args.clamp_percentile is not None and not (0 < args.clamp_percentile < 50):
        parser.error("--clamp-percentile must be between 1 and 49.")

    print(f"SEED: {torch.seed()}")
    # Set fixed seed for reproducibility
    # SEED = 5851323064657694470 # 2356925178268698826
    # random.seed(SEED)
    # # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        # torch.cuda.manual_seed_all(SEED)
    # elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    #     device = torch.device("mps")
    #     # Add MPS seeding if needed
    else:
        device = torch.device("cpu")

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # print(f"Using fixed seed: {SEED} on device: {device}")

    # # Create a generator for the DataLoader
    # g = torch.Generator()
    # g.manual_seed(SEED)

    # Load dataset
    if 'MNIST' in args.input_dir:
        model = load_model(args.model_path, device=device) # Pass device
        dataloader = get_mnist_data(batch_size=args.batch_size, generator=g) # Pass generator
    elif 'ImageNet' in args.input_dir:
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        # Modify get_imagenet_data to accept generator if needed
        config = argparse.Namespace()
        config.input_dir = args.input_dir
        config.batch_size = args.batch_size
        # config.generator = g
        config.subset_size = 0
        dataloader = get_imagenet_data(config) # Pass config with generator
    else:
        print(f"Error: Unknown dataset type based on input_dir: {args.input_dir}")
        return

    model = model.to(device) # Ensure model is on the correct device
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device) # Move data to device

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # --- Load Combined LRP Parameters and Weights ---
    lrp_params_paths = {
        'sensitivity': args.params_sensitivity,
        'sparseness': args.params_sparseness,
        'faithfulness': args.params_faithfulness
    }
    lrp_weights = {
        'sensitivity': args.weight_sensitivity,
        'sparseness': args.weight_sparseness,
        'faithfulness': args.weight_faithfulness
    }

    loaded_lrp_params = {}
    for name, filepath in lrp_params_paths.items():
        if filepath:
            try:
                path = Path(filepath)
                if not path.exists():
                    print(f"Warning: Parameter file not found: {filepath}. Skipping '{name}'.")
                    continue
                with open(path, 'r') as f:
                    params_data = json.load(f)
                loaded_lrp_params[name] = params_data # Store raw data for now
                print(f"Successfully loaded parameters for '{name}' from {filepath}")
            except json.JSONDecodeError:
                print(f"Warning: Error decoding JSON from file: {filepath}. Skipping '{name}'.")
            except Exception as e:
                print(f"Warning: Error loading parameters for '{name}' from {filepath}: {e}. Skipping.")
        else:
             # Only print info if the weight is non-zero, otherwise it doesn't matter
             if lrp_weights.get(name, 0) != 0:
                 print(f"Info: No parameter file specified for '{name}', skipping.")


    # Check if any parameters were loaded and have non-zero weights
    active_lrp_params = {k: v for k, v in loaded_lrp_params.items() if lrp_weights.get(k, 0) != 0}
    if not active_lrp_params:
        print("Error: No LRP parameter files with non-zero weights were successfully loaded or specified. Cannot compute Combined LRP Difference.")
        # Decide whether to exit or continue without Combined LRP
        # return # Option: Exit if no LRP params loaded

    # Generate visualizations comparing XAI methods
    compare_xai_methods(
        model=model,
        images=images,
        labels=labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        loaded_lrp_params=loaded_lrp_params, # Pass all loaded, filtering happens inside
        lrp_weights=lrp_weights,
        diff_top_n=args.diff_top_n,      # Pass the new argument
        overlay=args.overlay,
        clamp_percentile=args.clamp_percentile,
        device=device
    )

    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()