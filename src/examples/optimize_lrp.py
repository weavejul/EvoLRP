import torch
import numpy as np
import matplotlib.pyplot as plt
from optim import CMAESOptimizer
from evaluation import create_fitness_function
from datasets import get_mnist_data
from models import load_model
from lrp import UniformRuleLRP
from utils import plot_heatmap

def optimize_lrp_parameters(criterion='sparseness', num_generations=20):
    """
    Example function demonstrating how to optimize LRP parameters using
    different evaluation criteria.
    
    Args:
        criterion: Which evaluation criterion to use
                   ('sparseness', 'pixel_flipping', 'average_sensitivity')
        num_generations: Number of generations for the CMA-ES optimization
    """
    # Load data and model
    batch_size = 4  # Small batch for demonstration
    dataloader = get_mnist_data(batch_size=batch_size)
    images, labels = next(iter(dataloader))
    print(f"Processing batch of {images.shape[0]} images")
    
    # Load MNIST model
    model = load_model('models/mnist_cnn.pth', device='cpu')
    model.eval()
    
    # Count number of layers
    num_layers = len(list(model.features)) + len(list(model.classifier))
    print(f"Optimizing {num_layers} alpha parameters")
    
    # Define initial parameters (one per layer)
    initial_params = torch.ones(num_layers) * 1.5
    rule_type = 'alpha_beta'
    
    # Compute initial relevance maps for reference
    lrp_initial = UniformRuleLRP(model, rule_type=rule_type, rule_params=initial_params)
    with torch.no_grad():
        initial_relevance = lrp_initial.compute_relevance(images)
    
    # Create fitness function with the specified criterion
    fitness_function = create_fitness_function(
        model=model,
        images=images,
        labels=labels,
        criterion=criterion,
        rule_type=rule_type,
        num_steps=5 if criterion in ['pixel_flipping', 'average_sensitivity'] else None
    )
    
    # Configure optimizer
    optimizer = CMAESOptimizer(
        num_params=num_layers,
        init_sigma=0.3,
        population_size=20
    )
    
    # Run optimization
    for gen in range(num_generations):
        optimizer.step(fitness_function)
        if gen % 5 == 0 or gen == num_generations - 1:
            print(f"Generation {gen}/{num_generations}")
    
    # Get optimized parameters
    best_params = optimizer.best_solution()
    
    # Apply constraints to the parameters
    constrained_best = []
    for p in best_params:
        constrained_best.append(max(1.0, abs(p) + 1.0))
    constrained_best = torch.tensor(constrained_best)
    
    print(f"Optimized alpha parameters:")
    for i, alpha in enumerate(constrained_best):
        print(f"  Layer {i+1}: {alpha:.4f}")
    
    # Compute optimized relevance maps
    lrp_optimized = UniformRuleLRP(model, rule_type=rule_type, rule_params=constrained_best)
    with torch.no_grad():
        optimized_relevance = lrp_optimized.compute_relevance(images)
    
    # Visualize results
    plt.figure(figsize=(12, 3 * images.shape[0]))
    for i in range(images.shape[0]):
        # Input image
        plt.subplot(images.shape[0], 3, i*3 + 1)
        plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title(f"Input (Label: {labels[i].item()})")
        plt.axis('off')
        
        # Initial relevance
        plt.subplot(images.shape[0], 3, i*3 + 2)
        plot_heatmap(initial_relevance[i], show=False)
        plt.title("Initial LRP")
        plt.axis('off')
        
        # Optimized relevance
        plt.subplot(images.shape[0], 3, i*3 + 3)
        plot_heatmap(optimized_relevance[i], show=False)
        plt.title(f"Optimized ({criterion})")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Demonstrate with different criteria
    print("Optimizing LRP parameters for maximum sparseness")
    optimize_lrp_parameters(criterion='sparseness', num_generations=15)
    
    print("\nOptimizing LRP parameters for maximum pixel flipping effect")
    optimize_lrp_parameters(criterion='pixel_flipping', num_generations=15)
    
    print("\nOptimizing LRP parameters for maximum average sensitivity")
    optimize_lrp_parameters(criterion='average_sensitivity', num_generations=15) 