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

def compare_xai_methods(
    model,
    images,
    labels,
    output_dir=None,
    batch_size=10,
    optimized_params=None
):
    """
    Compare visualizations using different XAI methods (GradCAM, Lime, and IntegratedGradients).
    
    Args:
        model: The neural network model
        images: Input images to visualize (B, C, H, W)
        labels: Labels for the images
        output_dir: Directory to save visualizations
        batch_size: Number of examples to visualize
    """
    # Limit number of examples
    batch_size = min(batch_size, len(images))
    
    # Create feature masks for each image in the batch for LIME
    feature_masks = []
    for img in images[:batch_size]:
        # Convert single image to numpy format (C, H, W) -> (H, W, C)
        x_numpy = img.cpu().permute(1, 2, 0).numpy()
        # Handle both grayscale and RGB images
        if x_numpy.shape[2] == 1:
            x_numpy = x_numpy.squeeze(2)  # Remove channel dim for grayscale
        feature_mask = slic(x_numpy, n_segments=100, compactness=10, sigma=0.5)
        feature_masks.append(feature_mask)
    feature_masks = torch.tensor(np.stack(feature_masks))
    
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
                images[:batch_size],
                target=labels[:batch_size]
            )
            
            # Use a more sophisticated interpolation method
            gradcam_attr = LayerAttribution.interpolate(
                gradcam_attr, 
                images[:batch_size].shape[2:],
                interpolate_mode='bicubic'  # Use bicubic interpolation instead of default
            )

            # guided
            # guided_gradcam = GuidedGradCam(model, target_layer)
            # gradcam_attr = guided_gradcam.attribute(
            #     images[:batch_size],
            #     target=labels[:batch_size]
            # )

            attributions['GradCAM'] = gradcam_attr.detach().cpu().numpy()
        
        # LIME
        lime = Lime(model)
        lime_attr = lime.attribute(
            images[:batch_size],
            target=labels[:batch_size],
            feature_mask=feature_masks.unsqueeze(1),
            n_samples=40,
            perturbations_per_eval=16,
            show_progress=False
        )
        attributions['Lime'] = lime_attr.detach().cpu().numpy()
        
        # IntegratedGradients
        ig = IntegratedGradients(model)
        ig_attr, _ = ig.attribute(
            images[:batch_size],
            baselines=0,
            target=labels[:batch_size],
            method='gausslegendre',
            return_convergence_delta=True
        )
        attributions['IntegratedGradients'] = ig_attr.detach().cpu().numpy()

        # LRP
        lrp = UniformRuleLRP(model, rule_type='alpha_beta', rule_params=optimized_params)
        lrp_attr = lrp.compute_relevance(images[:batch_size])
        attributions['URO_LRP'] = lrp_attr.detach().cpu().numpy()
    
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
            
            # Handle different attribution formats
            if heatmap.ndim == 3:
                if heatmap.shape[0] == 3:  # Channel-first format (C, H, W)
                    heatmap = np.transpose(heatmap, (1, 2, 0))
                
                # For RGB attributions, use magnitude across channels
                if heatmap.shape[-1] == 3:
                    heatmap = np.linalg.norm(heatmap, axis=2)
            
            # Normalize for visualization
            vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
            if vmax == 0:  # Avoid division by zero
                vmax = 1.0
            vmin = -vmax
            
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
                
            plt.title(f"Image {i+1}")
            plt.axis('off')
            
            # Plot each XAI method - place in columns after the original image
            for idx, (method_name, attr) in enumerate(attributions.items()):
                plt.subplot(batch_size, num_cols, i * num_cols + idx + 2)
                
                heatmap = attr[i].squeeze()
                
                # Handle different attribution formats
                if heatmap.ndim == 3:
                    if heatmap.shape[0] == 3:  # Channel-first format (C, H, W)
                        heatmap = np.transpose(heatmap, (1, 2, 0))
                    
                    # For RGB attributions, use magnitude across channels
                    if heatmap.shape[-1] == 3:
                        heatmap = np.linalg.norm(heatmap, axis=2)
                
                # Normalize for visualization
                vmax = max(abs(np.max(heatmap)), abs(np.min(heatmap)))
                if vmax == 0:  # Avoid division by zero
                    vmax = 1.0
                vmin = -vmax
                
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
    parser.add_argument('--lrp_params', type=str, default='optimized_alphas_in.json', help='Path to the LRP parameters')
    parser.add_argument('--batch_size', type=int, default=5, help='Number of examples to visualize')
    parser.add_argument('--input_dir', type=str, default='/home/neuro/XAIProject/data/ImageNet', help='Input directory for images')
    
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

    with open(args.lrp_params, 'r') as f:
        optimized_params = torch.tensor(json.load(f))
    
    # Generate visualizations comparing all XAI methods
    compare_xai_methods(
        model=model,
        images=images,
        labels=labels,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        optimized_params=optimized_params
    )
    
    print(f"Visualizations saved to {args.output_dir}")


if __name__ == "__main__":
    main()