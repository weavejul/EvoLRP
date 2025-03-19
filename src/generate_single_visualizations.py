import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from matplotlib.colors import Normalize

from lrp import UniformRuleLRP
from models import load_model
from datasets import get_mnist_data
from utils import plot_heatmap

import quantus
from skimage.segmentation import slic
from captum.attr import LayerAttribution, LayerGradCam, Lime


def compare_lrp_visualizations(
    model,
    images,
    labels,
    initial_params,
    optimized_params,
    output_dir=None,
    num_examples=10,
    rule_type='alpha_beta',
    xai_method=None
):
    """
    Compare LRP visualizations using initial and optimized alpha parameters.
    
    Args:
        model: The neural network model
        images: Input images to visualize (B, C, H, W)
        labels: Labels for the images
        initial_params: Initial alpha parameters (tensor)
        optimized_params: Optimized alpha parameters (tensor)
        output_dir: Directory to save visualizations
        num_examples: Number of examples to visualize
        rule_type: Type of LRP rule ('alpha_beta', 'epsilon', etc.)
    """
    # Limit number of examples
    num_examples = min(num_examples, len(images))
    
    # Create LRP instances with both parameter sets
    if xai_method == 'IntegratedGradients':
        explain_func_kwargs = {
            "method": "IntegratedGradients"
        }
    elif xai_method == 'Lime':
        # Create feature masks for each image in the batch
        feature_masks = []
        for img in images:
            # Convert single image to numpy format (C, H, W) -> (H, W, C)
            x_numpy = img.cpu().permute(1, 2, 0).numpy()
            feature_mask = slic(x_numpy, n_segments=100, compactness=10, sigma=0.5)
            feature_masks.append(feature_mask)
        
        # Stack all feature masks and move to device
        feature_masks = torch.tensor(np.stack(feature_masks))
        lime = Lime(model)



    lrp_optimized = UniformRuleLRP(model, rule_type=rule_type, rule_params=optimized_params)
    
    # Compute LRP relevance maps
    with torch.no_grad():
        if xai_method == 'Lime':
            initial_relevance = lime.attribute(
                        images[:num_examples],
                        target=labels[:num_examples],
                        feature_mask=feature_masks.unsqueeze(1),
                        n_samples=40,
                        perturbations_per_eval=16,
                        show_progress=True
                    )       
        elif xai_method == 'IntegratedGradients':
            initial_relevance = quantus.explain(model, images[:num_examples], labels[:num_examples], kwargs=explain_func_kwargs)
        initial_relevance = initial_relevance.detach().cpu().numpy()
        optimized_relevance = lrp_optimized.compute_relevance(images[:num_examples])
    
    # Create visualizations
    for i in range(num_examples):
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Input Image (Label: {labels[i].item()})")
        plt.axis('off')
        
        # Initial LRP heatmap
        plt.subplot(1, 3, 2)
        
        # Use bidirectional colormap centered at zero
        heatmap = initial_relevance[i].squeeze()
        vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
        vmin = -vmax
        
        plt.imshow(heatmap, cmap='bwr', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f"Initial Parameters")
        plt.axis('off')
        
        # Optimized LRP heatmap
        plt.subplot(1, 3, 3)
        
        # Use bidirectional colormap centered at zero
        heatmap = optimized_relevance[i].squeeze().cpu().numpy()
        vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
        vmin = -vmax
        
        plt.imshow(heatmap, cmap='bwr', vmin=vmin, vmax=vmax)
        plt.colorbar()
        plt.title(f"Optimized Parameters")
        plt.axis('off')
        
        plt.tight_layout()
        
        # Save or display the figure
        if output_dir:
            output_path = Path(output_dir) / f"lrp_comparison_example_{i}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    # Create a summary image with multiple examples in a grid
    if num_examples > 1:
        # Calculate grid dimensions
        grid_size = min(5, num_examples)  # Max 5 examples per row
        rows = (num_examples + grid_size - 1) // grid_size
        
        plt.figure(figsize=(grid_size * 5, rows * 5))
        
        for i in range(num_examples):
            # Original image
            plt.subplot(rows, grid_size * 3, i * 3 + 1)
            plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"Image {i+1}")
            plt.axis('off')
            
            # Initial LRP heatmap
            plt.subplot(rows, grid_size * 3, i * 3 + 2)
            heatmap = initial_relevance[i].squeeze()
            vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
            vmin = -vmax
            plt.imshow(heatmap, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.title(f"Initial")
            plt.axis('off')
            
            # Optimized LRP heatmap
            plt.subplot(rows, grid_size * 3, i * 3 + 3)
            heatmap = optimized_relevance[i].squeeze().cpu().numpy()
            vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
            vmin = -vmax
            plt.imshow(heatmap, cmap='bwr', vmin=vmin, vmax=vmax)
            plt.title(f"Optimized")
            plt.axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "lrp_comparison_summary.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    # Print parameter comparison
    print("\n===== Parameter Comparison =====")
    print("Layer\tInitial\tOptimized\tChange")
    print("-------------------------------------")


def main():
    parser = argparse.ArgumentParser(description='Compare LRP visualizations with different alpha values')
    parser.add_argument('--model_path', type=str, default='models/mnist_cnn.pth', help='Path to the model')
    parser.add_argument('--optimized_params', type=str, required=True, help='Path to optimized alpha parameters (JSON)')
    parser.add_argument('--output_dir', type=str, default='lrp_comparisons', help='Output directory for visualizations')
    parser.add_argument('--num_examples', type=int, default=10, help='Number of examples to visualize')
    parser.add_argument('--rule_type', type=str, default='alpha_beta', help='LRP rule type')
    parser.add_argument('--xai_method', type=str, default=None, help='XAI method to use')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model_path, device='cpu')
    model.eval()
    
    # Load dataset
    dataloader = get_mnist_data(batch_size=args.num_examples)
    images, labels = next(iter(dataloader))
    
    # Load parameters from JSON files
    # with open(args.initial_params, 'r') as f:
    #     initial_params = torch.tensor(json.load(f))
    
    with open(args.optimized_params, 'r') as f:
        optimized_params = torch.tensor(json.load(f))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Generate visualizations
    compare_lrp_visualizations(
        model=model,
        images=images,
        labels=labels,
        initial_params=None,
        optimized_params=optimized_params,
        output_dir=args.output_dir,
        num_examples=args.num_examples,
        rule_type=args.rule_type,
        xai_method=args.xai_method
    )
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()