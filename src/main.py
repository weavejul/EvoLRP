import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from optim import CMAESOptimizer
from evaluation import pixel_flipping, average_sensitivity, sparseness, create_fitness_function, faithfulness_correlation
from datasets import get_mnist_data, get_imagenet_data
from models import ModelWrapper
from models import MNISTNet, load_model
from lrp import UniformRuleLRP
from utils import plot_heatmap
import argparse

def main(rule='alpha_beta', dataset='mnist', init_value=2.0):
    try:
        # Load dataset based on the specified type
        batch_size = 8
        
        if dataset == 'mnist':
            dataloader = get_mnist_data(batch_size=batch_size)
            # Load our MNIST model
            model = load_model('models/mnist_cnn.pth')
        else:  # ImageNet with VGG16
            # Create config for the data loader
            config = argparse.Namespace()
            config.input_dir = '../../data/ImageNet'  # Updated path relative to EvoLRP/src directory
            config.batch_size = batch_size
            config.resize = 224
            config.subset_size = 100  # Optional: Use a subset for faster testing
            dataloader = get_imagenet_data(config)
            
            # Load pretrained VGG16 with updated API
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            model.eval()
        
        # Get a batch of data
        images, labels = next(iter(dataloader))
        print(f"Processing batch of {images.shape[0]} images with labels: {labels}")
        
        # Ensure model is in eval mode
        model.eval()
        
        # Get the device and ensure model and data are on the same device
        device = next(model.parameters()).device  # Get device model is on
        images = images.to(device)  # Move images to the same device
        labels = labels.to(device)  # Move labels as well
        
        # Define CMA-ES optimizer (one parameter per layer)
        if dataset == 'mnist':
            num_layers = len(list(model.features)) + len(list(model.classifier))
        else:  # VGG16
            num_layers = len(list(model.features)) + len(list(model.classifier))
        
        print(f"Optimizing {num_layers} parameters")
        
        # Choose rule type - sets the appropriate parameters based on rule
        rule_type = rule  # Can be 'epsilon', 'alpha_beta', or 'gamma'
        
        # Simplified initialization - use the same value for all parameters
        if rule_type == 'alpha_beta':
            # For alpha-beta rule, all layers start with the same alpha value
            initial_params = torch.ones(num_layers) * init_value
            print(f"Initial alpha-beta parameters: {initial_params.tolist()}")
        
        elif rule_type == 'epsilon':
            # For epsilon rule, all layers start with the same epsilon fraction
            initial_params = torch.ones(num_layers) * init_value
            print(f"Initial epsilon parameters (as fraction of std): {initial_params.tolist()}")
        
        else:
            raise ValueError(f"Unsupported rule type: {rule_type}")
        
        # Create initial LRP instance
        lrp_initial = UniformRuleLRP(model, rule_type=rule_type, rule_params=initial_params)
        
        # Compute and visualize initial relevance map
        with torch.no_grad():
            initial_relevance = lrp_initial.compute_relevance(images)
            print("Initial relevance map shape:", initial_relevance.shape)
            
            # Get initial sparseness scores for reference
            initial_sparseness_scores = []
            
            # for i in range(images.shape[0]):
            # Calculate sparseness for this image
            # sparseness(initial_relevance)
            kwargs = {
                'subset_size': 10, # check this
            }
            initial_sparseness_scores = pixel_flipping(initial_relevance, model, images, labels)#sparseness(initial_relevance)
            #faithfulness_correlation(initial_relevance, model, images, labels, **kwargs)  # Get individual relevance map but keep batch dim
            # initial_sparseness_scores.append(sparse_score)
            
            print(f"Initial sparseness scores: {initial_sparseness_scores}")
            print(f"Average initial sparseness score: {sum(initial_sparseness_scores)/len(initial_sparseness_scores)}")
            
            # Plot the first image and its relevance map
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            # For ImageNet RGB images, we need to denormalize and use different display method
            if dataset == 'imagenet':
                # Denormalize ImageNet images for visualization
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                img_to_show = images[0].cpu() * std.cpu() + mean.cpu()
                img_to_show = img_to_show.permute(1, 2, 0).numpy()  # Convert to HWC format
                plt.imshow(img_to_show)
            else:
                # For MNIST grayscale images
                plt.imshow(images[0].squeeze().cpu().numpy(), cmap='gray')
            plt.title(f"Input Image (Label: {labels[0].item()})")
            
            plt.subplot(1, 2, 2)
            plot_heatmap(initial_relevance[0], show=False)
            plt.title(f"Initial LRP ({rule_type})")
            plt.tight_layout()
            plt.show()
        
        # Track parameters and fitness over generations
        all_params = []
        all_fitness = []
        best_fitness_so_far = -float('inf')
        patience = 10  # Early stopping patience
        no_improvement = 0
        
        # Create fitness function using our new utility
        fitness_function = create_fitness_function(
            model=model,
            images=images,
            labels=labels,  # Provide labels even though not needed for sparseness
            criterion='pixel_flipping',
            rule_type=rule_type,
            weights=(0.7, 0.3)  # 70% weight on mean, 30% on min score
        )
        
        # Configure CMA-ES with better settings
        optimizer = CMAESOptimizer(num_params=len(initial_params), 
                                   init_sigma=0.3,  # Smaller sigma for more precise search
                                   population_size=100)  # Reasonable population size
        
        # Run optimization
        num_generations = 30  # Fewer generations since computation is expensive
        for gen in range(num_generations):
            optimizer.step(fitness_function)
            if gen % 5 == 0 or gen == num_generations-1:  # Print less frequently
                print(f"Generation {gen}")
                optimizer.es.disp()
                
            # Track best parameters and fitness after each generation
            best_params = optimizer.best_solution()
            best_fitness = -optimizer.es.result.fbest  # Negate back to positive
            all_params.append(best_params.copy())
            all_fitness.append(best_fitness)
            
            # Early stopping check
            if best_fitness > best_fitness_so_far:
                best_fitness_so_far = best_fitness
                no_improvement = 0
            else:
                no_improvement += 1
                
            if no_improvement >= patience:
                print(f"Early stopping at generation {gen} due to no improvement for {patience} generations")
                break
        
        # Retrieve best parameters
        best_params = optimizer.best_solution()
        print(f"Raw optimized parameters:", best_params)
        
        # Apply constraints based on rule type
        if rule_type == 'alpha_beta':
            # Apply constraints to get actual alpha values used
            constrained_best = []
            for i, p in enumerate(best_params):
                # Set minimum alpha values based on layer type
                min_val = 1.0
                
                # Constrain the parameter with layer-specific minimum
                constrained_best.append(max(min_val, abs(p) + 1.0))
                
            constrained_best = torch.tensor(constrained_best)
            print(f"Constrained alpha parameters:", constrained_best.tolist())
            print(f"Corresponding beta parameters:", (constrained_best - 1.0).tolist())
        
        elif rule_type == 'epsilon':
            # For epsilon rule, ensure values are positive and reasonable
            constrained_best = []
            for i, p in enumerate(best_params):
                # Epsilon must be positive
                constrained_best.append(max(0.01, abs(p)))
                
            constrained_best = torch.tensor(constrained_best)
            print(f"Constrained epsilon parameters (as fraction of std):", constrained_best.tolist())
        
        # Visualize LRP for best parameters on each image in the batch
        lrp_best = UniformRuleLRP(model, rule_type=rule_type, rule_params=constrained_best)
        with torch.no_grad():
            best_relevance = lrp_best.compute_relevance(images)
            print("Best relevance map shape:", best_relevance.shape)
            
            # Calculate optimized sparseness scores
            optimized_sparseness_scores = []
            
            for i in range(images.shape[0]):
                # Calculate sparseness for this image
                sparse_score = sparseness(
                    best_relevance  # Get individual relevance map but keep batch dim
                )
                optimized_sparseness_scores.append(sparse_score)
            
            print(f"Optimized sparseness scores: {optimized_sparseness_scores}")
            
            print(f"Average optimized sparseness score: {sum(optimized_sparseness_scores)/len(optimized_sparseness_scores)}")
            print(f"Sparseness Improvement: {sum(optimized_sparseness_scores)/len(optimized_sparseness_scores) - sum(initial_sparseness_scores)/len(initial_sparseness_scores)}")
            
            # Create a grid of images and their optimized relevance maps
            max_images = min(4, images.shape[0])
            plt.figure(figsize=(15, 5 * max_images))
            
            for i in range(max_images):
                # Original image
                plt.subplot(max_images, 3, i*3 + 1)
                
                if dataset == 'imagenet':
                    # Denormalize ImageNet images
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                    img_to_show = images[i].cpu() * std.cpu() + mean.cpu()
                    img_to_show = img_to_show.permute(1, 2, 0).numpy()
                    plt.imshow(img_to_show)
                else:
                    plt.imshow(images[i].squeeze().cpu().numpy(), cmap='gray')
                    
                plt.title(f"Image {i+1} (Label: {labels[i].item()})")
                
                # Initial relevance map
                plt.subplot(max_images, 3, i*3 + 2)
                plot_heatmap(initial_relevance[i], show=False)
                plt.title(f"Initial (Sparseness: {initial_sparseness_scores[i]:.3f})")
                
                # Optimized relevance map
                plt.subplot(max_images, 3, i*3 + 3)
                plot_heatmap(best_relevance[i], show=False)
                plt.title(f"Optimized (Sparseness: {optimized_sparseness_scores[i]:.3f})")
            
            plt.tight_layout()
            plt.show()
            
        # Plot parameter evolution
        all_params = np.array(all_params)
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot fitness evolution
        axs[0].plot(all_fitness)
        axs[0].set_title('Fitness Evolution (higher is better)')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel('Fitness (avg across batch)')
        
        # Plot parameter evolution
        for i in range(len(initial_params)):
            axs[1].plot(all_params[:, i], label=f'Layer {i+1}')
        axs[1].set_title(f'Parameter Evolution ({rule_type} rule)')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Raw Parameter Value')
        axs[1].legend()
        
        plt.tight_layout()
        plt.show()
            
    except Exception as e:
        print(f"Error in main: {e}")
        raise

if __name__ == "__main__":
    # Run with alpha-beta rule by default on MNIST
    # To run with epsilon rule, use: main(rule='epsilon')
    # To run with ImageNet and VGG16, use: main(dataset='imagenet')
    # To specify initialization value: main(init_value=2.0)
    main(rule='alpha_beta', dataset='imagenet', init_value=1.5)