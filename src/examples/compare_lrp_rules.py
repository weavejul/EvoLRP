import torch
import numpy as np
import matplotlib.pyplot as plt
from datasets import get_mnist_data
from models import load_model
from lrp import UniformRuleLRP
from utils import plot_heatmap
from evaluation import sparseness, pixel_flipping, average_sensitivity

def compare_lrp_rules(num_samples=4):
    """
    Compare different LRP rule types (alpha-beta vs epsilon) on MNIST examples.
    """
    # Load dataset
    batch_size = num_samples
    dataloader = get_mnist_data(batch_size=batch_size)
    images, labels = next(iter(dataloader))
    print(f"Analyzing {images.shape[0]} images with labels: {labels}")
    
    # Load model
    model = load_model('models/mnist_cnn.pth', device='cpu')
    model.eval()
    
    # Count number of layers
    num_layers = len(list(model.features)) + len(list(model.classifier))
    print(f"Model has {num_layers} layers")
    
    # Define parameters for alpha-beta rule
    alpha_beta_params = torch.ones(num_layers) * 2.0  # Alpha = 2, Beta = 1
    
    # Define parameters for epsilon rule
    # Try different fractions of std
    epsilon_params = {
        'low': torch.ones(num_layers) * 0.1,     # 10% of std - more detailed
        'medium': torch.ones(num_layers) * 0.25,  # 25% of std - balanced
        'high': torch.ones(num_layers) * 0.5    # 50% of std - sparser
    }
    
    # Create LRP instances
    lrp_alpha_beta = UniformRuleLRP(model, rule_type='alpha_beta', rule_params=alpha_beta_params)
    lrp_epsilon_low = UniformRuleLRP(model, rule_type='epsilon', rule_params=epsilon_params['low'])
    lrp_epsilon_med = UniformRuleLRP(model, rule_type='epsilon', rule_params=epsilon_params['medium'])
    lrp_epsilon_high = UniformRuleLRP(model, rule_type='epsilon', rule_params=epsilon_params['high'])
    
    with torch.no_grad():
        # Compute relevance maps
        rel_alpha_beta = lrp_alpha_beta.compute_relevance(images)
        rel_epsilon_low = lrp_epsilon_low.compute_relevance(images)
        rel_epsilon_med = lrp_epsilon_med.compute_relevance(images)
        rel_epsilon_high = lrp_epsilon_high.compute_relevance(images)
        
        # Metrics dictionary to store results
        metrics = {
            'alpha_beta': [],
            'epsilon_low': [],
            'epsilon_med': [],
            'epsilon_high': []
        }
        
        # Calculate metrics for each image
        for i in range(images.shape[0]):
            for rule_name, rel_map in [
                ('alpha_beta', rel_alpha_beta),
                ('epsilon_low', rel_epsilon_low),
                ('epsilon_med', rel_epsilon_med),
                ('epsilon_high', rel_epsilon_high)
            ]:
                # Calculate sparseness
                sparse_score = sparseness(rel_map[i:i+1])
                
                # Store metrics
                metrics[rule_name].append({
                    'sparseness': sparse_score
                })
                
        # Create visualization
        plt.figure(figsize=(16, 4 * num_samples))
        
        for i in range(images.shape[0]):
            # Original image
            plt.subplot(num_samples, 5, i*5 + 1)
            plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"Image (Label: {labels[i].item()})")
            plt.axis('off')
            
            # Alpha-Beta LRP
            plt.subplot(num_samples, 5, i*5 + 2)
            plot_heatmap(rel_alpha_beta[i], show=False)
            plt.title(f"Alpha-Beta LRP\nSparseness: {metrics['alpha_beta'][i]['sparseness']:.3f}")
            plt.axis('off')
            
            # Epsilon LRP (low)
            plt.subplot(num_samples, 5, i*5 + 3)
            plot_heatmap(rel_epsilon_low[i], show=False)
            plt.title(f"Epsilon (0.1 std)\nSparseness: {metrics['epsilon_low'][i]['sparseness']:.3f}")
            plt.axis('off')
            
            # Epsilon LRP (medium)
            plt.subplot(num_samples, 5, i*5 + 4)
            plot_heatmap(rel_epsilon_med[i], show=False)
            plt.title(f"Epsilon (0.25 std)\nSparseness: {metrics['epsilon_med'][i]['sparseness']:.3f}")
            plt.axis('off')
            
            # Epsilon LRP (high)
            plt.subplot(num_samples, 5, i*5 + 5)
            plot_heatmap(rel_epsilon_high[i], show=False)
            plt.title(f"Epsilon (0.5 std)\nSparseness: {metrics['epsilon_high'][i]['sparseness']:.3f}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Print average metrics
        print("\nAverage Metrics:")
        print("-" * 50)
        print(f"Rule Type            | Sparseness")
        print("-" * 50)
        for rule_name in metrics:
            avg_sparseness = np.mean([m['sparseness'] for m in metrics[rule_name]])
            print(f"{rule_name.ljust(20)} | {avg_sparseness:.4f}")

if __name__ == "__main__":
    compare_lrp_rules(num_samples=4) 