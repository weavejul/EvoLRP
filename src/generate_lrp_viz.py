import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import json
from matplotlib.colors import Normalize

from lrp import UniformRuleLRP
from models import load_model
from datasets import get_mnist_data, get_imagenet_data
from utils import plot_heatmap

import quantus
from skimage.segmentation import slic
from captum.attr import LayerAttribution, LayerGradCam, Lime, IntegratedGradients, GuidedGradCam
from torchvision.models import vgg16, VGG16_Weights

torch.manual_seed(7119481744208154007) # 5851323064657694470, 7119481744208154007

def compare_xai_methods(
    model,
    images,
    labels,
    output_dir=None,
    batch_size=10,
    optimized_params_ab=None,
    optimized_params_epsilon=None,
    optimized_params_gamma=None,
    optimized_params_four=None,
    overlay=False,  # New flag to control overlay visualization
    output_labels=None
):
    """
    Compare visualizations using different XAI methods (GradCAM, Lime, and IntegratedGradients).
    
    Args:
        model: The neural network model
        images: Input images to visualize (B, C, H, W)
        labels: Labels for the images
        output_dir: Directory to save visualizations
        batch_size: Number of examples to visualize
        optimized_params: Parameters for LRP
        overlay: If True, overlay heatmaps on original images instead of showing them separately
    """
    # Limit number of examples
    batch_size = min(batch_size, len(images))
    attributions = {}
    
    try:
        import json
        with open('/home/neuro/XAIProject/data/imagenet_class_index.json') as f:
            class_idx = json.load(f)
        idx_to_labels = {int(k): v[1] for k, v in class_idx.items()}
    except:
        # Fallback if file not found
        idx_to_labels = None
        print("Warning: ImageNet class labels file not found. Using numeric labels only.")
    # Compute attributions for each method
    with torch.no_grad():
        lrp_1 = UniformRuleLRP(model, rule_type='alpha_beta', rule_params=optimized_params_ab)
        lrp_attr_1 = lrp_1.compute_relevance(images[:batch_size], labels[:batch_size])
        attributions[output_labels[0]] = lrp_attr_1.detach().cpu().numpy()

        lrp_2 = UniformRuleLRP(model, rule_type='alpha_beta', rule_params=optimized_params_epsilon)
        lrp_attr_2 = lrp_2.compute_relevance(images[:batch_size], labels[:batch_size])
        attributions[output_labels[1]] = lrp_attr_2.detach().cpu().numpy()

        lrp_3 = UniformRuleLRP(model, rule_type='alpha_beta', rule_params=optimized_params_gamma)
        lrp_attr_3 = lrp_3.compute_relevance(images[:batch_size], labels[:batch_size])
        attributions[output_labels[2]] = lrp_attr_3.detach().cpu().numpy()

        lrp_4 = UniformRuleLRP(model, rule_type='alpha_beta', rule_params=optimized_params_four)
        lrp_attr_4 = lrp_4.compute_relevance(images[:batch_size], labels[:batch_size])
        attributions[output_labels[3]] = lrp_attr_4.detach().cpu().numpy()
    
    for method_name, attr in attributions.items():
       min_val = np.min(attr)
       max_val = np.max(attr)
       print(f"{method_name}: min={min_val}, max={max_val}")
    # Create visualizations for each example
    for i in range(batch_size):
        plt.figure(figsize=(15, 10))
        
        # Original image
        plt.subplot(1, 5, 1)
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
            
        plt.title(f"Input Image (Label: {labels[i].item()})")
        plt.axis('off')
        
        # Plot each XAI method
        for idx, (method_name, attr) in enumerate(attributions.items(), 2):
            plt.subplot(1, 5, idx)
                            
            heatmap = attr[i].squeeze()
            heatmap = np.mean(heatmap, axis=0)

            # Normalize for visualization
            # print(np.min(heatmap), np.max(heatmap))
            vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
            if vmax == 0:  # Avoid division by zero
                vmax = 1.0
            vmin = -vmax
            print(f"{method_name}: vmin={vmin}, vmax={vmax}")

            plt.imshow(heatmap, cmap='bwr', vmin=vmin, vmax=vmax)
            
            plt.colorbar()
            plt.title(method_name)
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
                
            if idx_to_labels is not None and labels[i].item() in idx_to_labels:
                label_text = f"Label: {idx_to_labels[labels[i].item()]}"
            else:
                label_text = f"Label: {labels[i].item()}"
            plt.title(f"{label_text}")
            plt.axis('off')
            
            # Plot each XAI method - place in columns after the original image
            for idx, (method_name, attr) in enumerate(attributions.items()):
                plt.subplot(batch_size, num_cols, i * num_cols + idx + 2)
                
                heatmap = attr[i].squeeze()
                heatmap = np.mean(heatmap, axis=0)

                # Normalize for visualization
                vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
                if vmax == 0:  # Avoid division by zero
                    vmax = 1.0
                vmin = -vmax
                
                if overlay:
                    # Create a colormap for the heatmap
                    cmap = plt.cm.bwr
                    norm = Normalize(vmin=vmin, vmax=vmax)
                    
                    # First display the original image
                    if img_display.shape[2] == 1:
                        plt.imshow(img_display.squeeze(2), cmap='gray')
                    else:
                        plt.imshow(img_display)
                    
                    # Then overlay the heatmap using the colormap and normalization
                    # This avoids creating RGB values outside the valid range
                    plt.imshow(heatmap, cmap='bwr', alpha=0.7, vmin=vmin, vmax=vmax)
                else:
                    # Original visualization without overlay
                    plt.imshow(heatmap, cmap='bwr', vmin=vmin, vmax=vmax)
                
                plt.title(method_name)
                plt.axis('off')
        
        plt.tight_layout()
        
        if output_dir:
            output_path = Path(output_dir) / "xai_methods_comparison_summary.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare different XAI visualization methods')
    parser.add_argument('--model_path', type=str, default='models/mnist_cnn.pth', help='Path to the model')
    parser.add_argument('--output_dir', type=str, default='xai_comparisons', help='Output directory for visualizations')
    parser.add_argument('--lrp_params_ab', type=str, default='saved_params/opt_alpha_beta_sparseness,faithfulness.json', help='Path to the LRP parameters')
    parser.add_argument('--lrp_params_epsilon', type=str, default='saved_params/opt_alpha_beta_average_sensitivity,faithfulness.json', help='Path to the LRP parameters')
    parser.add_argument('--lrp_params_gamma', type=str, default='saved_params/opt_alpha_beta_sparseness,average_sensitivity.json', help='Path to the LRP parameters')
    parser.add_argument('--lrp_params_four', type=str, default='saved_params/opt_alpha_beta_faithfulness.json', help='Path to the LRP parameters')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of examples to visualize')
    parser.add_argument('--input_dir', type=str, default='/home/neuro/XAIProject/data/ImageNet', help='Input directory for images')
    parser.add_argument('--overlay', action='store_true', help='Overlay heatmaps on original images')
    parser.add_argument('--output_labels', type=str, default='faithfulness, avg_sensitivity, sparseness, pixel flipping', help='Path to the labels')
    
    args = parser.parse_args()
        
    # Load dataset
    if 'MNIST' in args.input_dir:
        # Load model
        model = load_model(args.model_path, device='cpu')
        dataloader = get_mnist_data(batch_size=args.batch_size)
    elif 'ImageNet' in args.input_dir:
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        dataloader = get_imagenet_data(args)
    
    model.eval()
    images, labels = next(iter(dataloader))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    with open(args.lrp_params_ab, 'r') as f:
        optimized_params_ab = torch.tensor(json.load(f))
    
    with open(args.lrp_params_epsilon, 'r') as f:
        optimized_params_epsilon = torch.tensor(json.load(f))
    
    with open(args.lrp_params_gamma, 'r') as f:
        optimized_params_gamma = torch.tensor(json.load(f))
    
    with open(args.lrp_params_four, 'r') as f:
        optimized_params_four = torch.tensor(json.load(f))

    output_labels = args.output_labels.split(',')
    # Generate visualizations comparing all XAI methods
    compare_xai_methods(
        model=model,
        images=images,
        labels=labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        optimized_params_ab=optimized_params_ab,
        optimized_params_epsilon=optimized_params_epsilon,
        optimized_params_gamma=optimized_params_gamma,
        optimized_params_four=optimized_params_four,
        overlay=args.overlay,
        output_labels=output_labels
    )
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()