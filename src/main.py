import torch
import torchvision.models as models
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Project-specific imports
from optim import CMAESOptimizer
from evaluation import pixel_flipping, average_sensitivity, sparseness, create_fitness_function, faithfulness_correlation
from datasets import get_mnist_data, get_imagenet_data
from models import load_model
from lrp import UniformRuleLRP
from utils import plot_heatmap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("EvoLRP")

# Define available LRP rules and datasets
AVAILABLE_RULES = ['alpha_beta', 'epsilon', 'gamma']
AVAILABLE_DATASETS = ['mnist', 'imagenet']
AVAILABLE_CRITERIA = ['sparseness', 'pixel_flipping', 'average_sensitivity', 'faithfulness']
AVAILABLE_DEVICES = ['cpu', 'cuda', 'mps']
AVAILABLE_BASELINES = ['zero', 'mean']
AVAILABLE_OPTIMIZATION_TYPES = ['UniformRule', 'UniformValue', 'UniformLayer', 'CompositeRV']

def parse_arguments():
    """Parse command line arguments for the EvoLRP framework."""
    parser = argparse.ArgumentParser(
        description="EvoLRP - Evolutionary optimization for Layer-wise Relevance Propagation parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset and model configuration
    parser.add_argument('--dataset', type=str, default='mnist', choices=AVAILABLE_DATASETS,
                        help=f'Dataset to use: {", ".join(AVAILABLE_DATASETS)}')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory (defaults to ../../data/ImageNet for ImageNet, built-in for MNIST)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to custom model (defaults to pretrained models)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing images')
    parser.add_argument('--subset_size', type=int, default=100,
                        help='Subset size for ImageNet (to limit computation)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size for ImageNet preprocessing (resize)')
    
    # LRP rule configuration
    parser.add_argument('--rule', type=str, default='alpha_beta', choices=AVAILABLE_RULES,
                        help=f'LRP rule type: {", ".join(AVAILABLE_RULES)}')
    parser.add_argument('--init_value', type=float, default=0.25,
                        help='Initial value for all rule parameters')
    parser.add_argument('--correct_class_only', action='store_true',
                        help='Calculate relevance only for the correct class instead of all logits')
    
    # Optimization parameters
    parser.add_argument('--criterion', type=str, default='sparseness', choices=AVAILABLE_CRITERIA,
                        help=f'Fitness criterion: {", ".join(AVAILABLE_CRITERIA)}')
    parser.add_argument('--generations', type=int, default=30,
                        help='Number of generations for optimization')
    parser.add_argument('--population_size', type=int, default=100,
                        help='Population size for CMA-ES optimizer')
    parser.add_argument('--init_sigma', type=float, default=0.3,
                        help='Initial sigma for CMA-ES optimizer')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience (generations)')
    parser.add_argument('--mean_weight', type=float, default=0.7,
                        help='Weight for mean score in fitness function')
    parser.add_argument('--min_weight', type=float, default=0.3,
                        help='Weight for minimum score in fitness function')
    
    # Evaluation parameters
    parser.add_argument('--num_steps', type=int, default=5,
                        help='Number of steps for pixel flipping and sensitivity analysis')
    parser.add_argument('--baseline', type=str, default='zero', choices=AVAILABLE_BASELINES,
                        help=f'Baseline for sensitivity analysis: {", ".join(AVAILABLE_BASELINES)}')
    
    # Output and visualization
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save results')
    parser.add_argument('--save_params', action='store_true',
                        help='Save optimized parameters to a JSON file')
    parser.add_argument('--no_plot', action='store_true',
                        help='Disable visualization plots')
    parser.add_argument('--save_plots', action='store_true',
                        help='Save visualization plots to timestamped folder')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default=None, choices=AVAILABLE_DEVICES,
                        help=f'Device to use: {", ".join(AVAILABLE_DEVICES)}. If None, will use CUDA if available.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use when device is cuda (e.g. 0, 1, etc)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.mean_weight + args.min_weight != 1.0:
        logger.warning(f"Mean weight ({args.mean_weight}) + min weight ({args.min_weight}) != 1.0. Normalizing.")
        total = args.mean_weight + args.min_weight
        args.mean_weight /= total
        args.min_weight /= total
    
    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            args.device = 'mps'
        else:
            args.device = 'cpu'
            
    # Set GPU device if using CUDA
    if args.device == 'cuda':
        if args.gpu_id >= torch.cuda.device_count():
            logger.warning(f"GPU {args.gpu_id} not available, falling back to GPU 0")
            args.gpu_id = 0
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU {args.gpu_id}")
    
    # Set default data directory for ImageNet
    if args.data_dir is None and args.dataset == 'imagenet':
        args.data_dir = '../../data/ImageNet'
        
    # Create output directory
    if args.save_params:
        os.makedirs(args.output_dir, exist_ok=True)
        
    # Create visualization output directory if saving plots
    if args.save_plots:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        # Include more context in folder name
        folder_name = f"{timestamp}_{args.dataset}_{args.rule}_{args.criterion}_gen{args.generations}_pop{args.population_size}"
        args.plot_dir = os.path.join('lrp_results', folder_name)
        os.makedirs(args.plot_dir, exist_ok=True)
        
        # Save run configuration
        config_file = os.path.join(args.plot_dir, 'run_config.txt')
        with open(config_file, 'w') as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"LRP Rule: {args.rule}\n")
            f.write(f"Criterion: {args.criterion}\n")
            f.write(f"Generations: {args.generations}\n")
            f.write(f"Population Size: {args.population_size}\n")
            f.write(f"Initial Sigma: {args.init_sigma}\n")
            f.write(f"Initial Value: {args.init_value}\n")
            f.write(f"Mean Weight: {args.mean_weight}\n")
            f.write(f"Min Weight: {args.min_weight}\n")
            f.write(f"Device: {args.device}\n")
            f.write(f"Batch Size: {args.batch_size}\n")
            f.write(f"Correct Class Only: {args.correct_class_only}\n")
            if args.dataset == 'imagenet':
                f.write(f"Subset Size: {args.subset_size}\n")
                f.write(f"Image Size: {args.image_size}\n")
            
        logger.info(f"Saving visualizations and configuration to {args.plot_dir}")
        
    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    return args
def load_data_and_model(args):
    """Load dataset and model based on configuration."""
    try:
        device = torch.device(args.device)
        logger.info(f"Using device: {device}")
        
        # Load dataset
        if args.dataset == 'mnist':
            dataloader = get_mnist_data(batch_size=args.batch_size)
            logger.info(f"Loaded MNIST dataset with batch size {args.batch_size}")
            
            # Load MNIST model
            if args.model_path:
                model = load_model(args.model_path)
                logger.info(f"Loaded custom MNIST model from {args.model_path}")
            else:
                model = load_model('models/mnist_cnn.pth')
                logger.info("Loaded default MNIST model")
        else:  # ImageNet with VGG16
            # Configure ImageNet loader
            config = argparse.Namespace()
            config.input_dir = args.data_dir
            config.batch_size = args.batch_size
            config.resize = args.image_size
            config.subset_size = args.subset_size
            
            try:
                dataloader = get_imagenet_data(config)
                logger.info(f"Loaded ImageNet dataset with batch size {args.batch_size} and subset size {args.subset_size}")
            except Exception as e:
                logger.error(f"Failed to load ImageNet dataset: {e}")
                logger.error(f"Please make sure the dataset is available at {args.data_dir}")
                sys.exit(1)
            
            # Load VGG16 model
            if args.model_path:
                logger.info(f"Loading custom model from {args.model_path}")
                try:
                    model = torch.load(args.model_path)
                except Exception as e:
                    logger.error(f"Failed to load custom model: {e}")
                    sys.exit(1)
            else:
                logger.info("Loading pretrained VGG16 model")
                model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        
        # Move model to device and set to eval mode
        model = model.to(device)
        model.eval()
        
        # Get a batch of data
        try:
            images, labels = next(iter(dataloader))
            logger.info(f"Processed batch of {images.shape[0]} images with labels: {labels.tolist()}")
        
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)
            
            return model, images, labels, device
            
        except StopIteration:
            logger.error("Failed to get data batch. Dataloader is empty.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Error loading data and model: {e}")
        raise

def count_layers(model, dataset):
    """Count number of layers for parameter optimization."""
    try:
        if dataset == 'mnist':
            num_layers = len(list(model.features)) + len(list(model.classifier))
        elif dataset == 'imagenet':  # VGG16
            num_layers = len(list(model.features)) + len(list(model.classifier))
        else:
            logger.error(f"Unsupported dataset: {dataset}")
            sys.exit(1)
            
        logger.info(f"Model has {num_layers} layers for optimization")
        return num_layers
    except Exception as e:
        logger.error(f"Error counting model layers: {e}")
        logger.error("This may indicate a model structure incompatibility.")
        sys.exit(1)

def initialize_parameters(args, num_layers):
    """Initialize LRP rule parameters."""
    rule_type = args.rule
    init_value = args.init_value
    
    if rule_type == 'alpha_beta':
        # For alpha-beta rule, all layers start with the same alpha value
        initial_params = torch.ones(num_layers) * init_value
        logger.info(f"Initialized alpha-beta parameters: all layers set to alpha={init_value}")
        
    elif rule_type == 'epsilon':
        # For epsilon rule, all layers start with the same epsilon fraction
        initial_params = torch.ones(num_layers) * init_value
        logger.info(f"Initialized epsilon parameters: all layers set to epsilon={init_value}")
        
    elif rule_type == 'gamma':
        # For gamma rule, all layers start with the same gamma value
        initial_params = torch.ones(num_layers) * init_value
        logger.info(f"Initialized gamma parameters: all layers set to gamma={init_value}")
        
    else:
        logger.error(f"Unsupported rule type: {rule_type}")
        sys.exit(1)
    
    return initial_params, rule_type
        
def evaluate_initial_relevance(model, images, labels, rule_type, initial_params, args):
    """Compute and evaluate initial relevance maps."""
    try:
        # Create initial LRP instance
        lrp_initial = UniformRuleLRP(model, rule_type=rule_type, rule_params=initial_params, 
                                   correct_class_only=args.correct_class_only)
        
        # Compute initial relevance map
        with torch.no_grad():
            initial_relevance = lrp_initial.compute_relevance(images, labels if args.correct_class_only else None)
            logger.info(f"Computed initial relevance maps with shape: {initial_relevance.shape}")
            
            # Get initial evaluation scores
            if args.criterion == 'sparseness':
                initial_scores = sparseness(initial_relevance)
            elif args.criterion == 'pixel_flipping':
                initial_scores = pixel_flipping(
                    initial_relevance, model, images, labels, num_steps=args.num_steps
                )
            elif args.criterion == 'average_sensitivity':
                initial_scores, _ = average_sensitivity(
                    initial_relevance, model, images, labels, 
                    num_steps=args.num_steps, baseline=args.baseline
                )
            elif args.criterion == 'faithfulness':
                initial_scores = faithfulness_correlation(
                    initial_relevance, model, images, labels, 
                    num_steps=args.num_steps, subset_size=10
                )
            
            logger.info(f"Initial {args.criterion} scores: {initial_scores}")
            average_score = sum(initial_scores) / len(initial_scores)
            logger.info(f"Average initial {args.criterion} score: {average_score:.4f}")
            
            return initial_relevance, initial_scores
            
    except Exception as e:
        logger.error(f"Error evaluating initial relevance: {e}")
        raise

def visualize_initial_results(images, initial_relevance, initial_scores, labels, dataset, rule_type, args, device):
    """Visualize initial input image and relevance map."""
    if args.no_plot and not args.save_plots:
        logger.info("Visualization disabled. Skipping initial visualization.")
        return
        
    try:
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
        
        # Add correct class info to title if enabled
        class_info = " (Correct Class Only)" if args.correct_class_only else ""
        plt.title(f"Initial LRP ({rule_type}{class_info}, Score: {initial_scores[0]:.3f})")
        plt.tight_layout()
        
        if args.save_plots:
            plt.savefig(os.path.join(args.plot_dir, 'initial_results.png'))
            logger.info("Saved initial results visualization")
            
        if not args.no_plot:
            plt.show()
        plt.close()
        
    except Exception as e:
        logger.error(f"Error visualizing initial results: {e}")
        logger.error("Continuing without visualization.")

def run_optimization(model, images, labels, rule_type, initial_params, args):
    """Run optimization process for LRP parameters."""
    try:
        # Create fitness function
        weights = (args.mean_weight, args.min_weight)
        fitness_function = create_fitness_function(
            model=model,
            images=images,
            labels=labels,
            criterion=args.criterion,
            rule_type=rule_type,
            weights=weights,
            baseline=args.baseline,
            num_steps=args.num_steps,
            correct_class_only=args.correct_class_only
        )
        
        # Configure CMA-ES optimizer
        optimizer = CMAESOptimizer(
            num_params=len(initial_params), 
            init_sigma=args.init_sigma,
            population_size=args.population_size
        )
        
        # Track parameters and fitness over generations
        all_params = []
        all_fitness = []
        best_fitness_so_far = -float('inf')
        no_improvement = 0
        
        # Run optimization
        logger.info(f"Starting optimization with {args.generations} generations")
        logger.info(f"Using {args.criterion} criterion with weights: mean={args.mean_weight}, min={args.min_weight}")
        
        for gen in range(args.generations):
            optimizer.step(fitness_function)
            
            # Print status periodically
            if gen % 5 == 0 or gen == args.generations - 1:
                logger.info(f"Generation {gen+1}/{args.generations}")
                optimizer.es.disp()
                
            # Track best parameters and fitness
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
                
            if no_improvement >= args.patience:
                logger.info(f"Early stopping at generation {gen+1} due to no improvement for {args.patience} generations")
                break
        
        logger.info("Optimization complete")
        return optimizer.best_solution(), all_params, all_fitness
        
    except Exception as e:
        logger.error(f"Error during optimization: {e}")
        raise

def constrain_parameters(best_params, rule_type):
    """Apply constraints to best parameters based on rule type."""
    constrained_best = []
    
    if rule_type == 'alpha_beta':
        # Apply constraints to get actual alpha values
        for i, p in enumerate(best_params):
            # Minimum alpha value
            min_val = 1.0
            # Constrain parameter with minimum
            constrained_best.append(max(min_val, abs(p) + 1.0))
                
        constrained_best = torch.tensor(constrained_best)
        logger.info(f"Constrained alpha parameters: {constrained_best.tolist()}")
        logger.info(f"Corresponding beta parameters: {(constrained_best - 1.0).tolist()}")
        
    elif rule_type == 'epsilon':
        # For epsilon rule, ensure values are positive and reasonable
        for i, p in enumerate(best_params):
            # Epsilon must be positive
            constrained_best.append(min(1.0, abs(p))) # max(0.01, abs(p))
                
        constrained_best = torch.tensor(constrained_best)
        logger.info(f"Constrained epsilon parameters: {constrained_best.tolist()}")
    
    elif rule_type == 'gamma':
        # For gamma rule, ensure values are positive
        # Values < 1 flatten the distribution, values > 1 sharpen it
        for i, p in enumerate(best_params):
            # Gamma should be positive and have a reasonable range
            constrained_best.append(max(0.1, min(5.0, abs(p))))
                
        constrained_best = torch.tensor(constrained_best)
        logger.info(f"Constrained gamma parameters: {constrained_best.tolist()}")
    
    return constrained_best
        
def evaluate_optimized_relevance(model, images, labels, rule_type, constrained_best, args):
    """Compute and evaluate optimized relevance maps."""
    try:
        # Create LRP with optimized parameters
        lrp_best = UniformRuleLRP(model, rule_type=rule_type, rule_params=constrained_best, 
                               correct_class_only=args.correct_class_only)
        
        # Compute optimized relevance map
        with torch.no_grad():
            best_relevance = lrp_best.compute_relevance(images, labels if args.correct_class_only else None)
            logger.info(f"Computed optimized relevance maps with shape: {best_relevance.shape}")
            
            # Get optimized evaluation scores
            if args.criterion == 'sparseness':
                optimized_scores = sparseness(best_relevance)
            elif args.criterion == 'pixel_flipping':
                optimized_scores = pixel_flipping(
                    best_relevance, model, images, labels, num_steps=args.num_steps
                )
            elif args.criterion == 'average_sensitivity':
                optimized_scores, _ = average_sensitivity(
                    best_relevance, model, images, labels, 
                    num_steps=args.num_steps, baseline=args.baseline
                )
            elif args.criterion == 'faithfulness':
                optimized_scores = faithfulness_correlation(
                    best_relevance, model, images, labels, 
                    num_steps=args.num_steps, subset_size=10
                )
            
            logger.info(f"Optimized {args.criterion} scores: {optimized_scores}")
            average_score = sum(optimized_scores) / len(optimized_scores)
            logger.info(f"Average optimized {args.criterion} score: {average_score:.4f}")
            
            return best_relevance, optimized_scores
            
    except Exception as e:
        logger.error(f"Error evaluating optimized relevance: {e}")
        raise

def visualize_comparison(images, initial_relevance, best_relevance, initial_scores, optimized_scores, 
                        labels, dataset, rule_type, args, device):
    """Visualize comparison between initial and optimized relevance maps."""
    if args.no_plot and not args.save_plots:
        logger.info("Visualization disabled. Skipping comparison visualization.")
        return
        
    try:
        # Create a grid of images and their optimized relevance maps
        max_images = min(4, images.shape[0])
        plt.figure(figsize=(15, 5 * max_images))
        
        # Add correct class info to title if enabled
        class_info = " (Correct Class Only)" if args.correct_class_only else ""
            
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
            plt.title(f"Initial{class_info} (Score: {initial_scores[i]:.3f})")
                
            # Optimized relevance map
            plt.subplot(max_images, 3, i*3 + 3)
            plot_heatmap(best_relevance[i], show=False)
            plt.title(f"Optimized{class_info} (Score: {optimized_scores[i]:.3f})")
            
        plt.tight_layout()
        
        if args.save_plots:
            plt.savefig(os.path.join(args.plot_dir, 'comparison_results.png'))
            logger.info("Saved comparison visualization")
            
        if not args.no_plot:
            plt.show()
        plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing comparison: {e}")
        logger.error("Continuing without visualization.")

def visualize_evolution(all_params, all_fitness, initial_params, rule_type, args):
    """Visualize parameter and fitness evolution over generations."""
    if args.no_plot and not args.save_plots:
        logger.info("Visualization disabled. Skipping evolution visualization.")
        return
        
    try:
        # Plot parameter evolution
        all_params_np = np.array(all_params)
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot fitness evolution
        axs[0].plot(all_fitness)
        axs[0].set_title('Fitness Evolution (higher is better)')
        axs[0].set_xlabel('Generation')
        axs[0].set_ylabel(f'Fitness ({args.criterion})')
        
        # Plot parameter evolution
        for i in range(len(initial_params)):
            axs[1].plot(all_params_np[:, i], label=f'Layer {i+1}')
        axs[1].set_title(f'Parameter Evolution ({rule_type} rule)')
        axs[1].set_xlabel('Generation')
        axs[1].set_ylabel('Raw Parameter Value')
        
        # Only show legend if we have a reasonable number of parameters
        if len(initial_params) <= 20:
            axs[1].legend()
        
        plt.tight_layout()
        
        if args.save_plots:
            plt.savefig(os.path.join(args.plot_dir, 'evolution.png'))
            logger.info("Saved evolution visualization")
            
        if not args.no_plot:
            plt.show()
        plt.close()
            
    except Exception as e:
        logger.error(f"Error visualizing evolution: {e}")
        logger.error("Continuing without visualization.")

def save_parameters(constrained_best, rule_type, args):
    """Save optimized parameters to JSON file."""
    if not args.save_params:
        return
        
    try:
        # Create output filename
        timestamp = Path(args.output_dir) / f"optimized_{rule_type}_{args.criterion}_{args.dataset}.json"
        
        # Prepare data to save
        params_data = {
            "rule_type": rule_type,
            "criterion": args.criterion,
            "dataset": args.dataset,
            "correct_class_only": args.correct_class_only,
            "parameters": constrained_best.tolist()
        }
        
        # Save to file
        with open(timestamp, 'w') as f:
            json.dump(params_data, f, indent=2)
            
        logger.info(f"Saved optimized parameters to {timestamp}")
        
    except Exception as e:
        logger.error(f"Error saving parameters: {e}")
        logger.error("Continuing without saving.")

def main():
    """Main function for EvoLRP optimization."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        logger.info(f"Running EvoLRP with {args.rule} rule on {args.dataset} dataset")
        if args.correct_class_only:
            logger.info("Relevance will be calculated for the correct class only")
        else:
            logger.info("Relevance will be calculated for all output neurons")
        
        # Load data and model
        model, images, labels, device = load_data_and_model(args)
        
        # Count layers for parameter optimization
        num_layers = count_layers(model, args.dataset)
        
        # Initialize parameters
        initial_params, rule_type = initialize_parameters(args, num_layers)
        
        # Evaluate initial relevance
        initial_relevance, initial_scores = evaluate_initial_relevance(
            model, images, labels, rule_type, initial_params, args
        )
        
        # Visualize initial results
        visualize_initial_results(
            images, initial_relevance, initial_scores, labels, args.dataset, rule_type, args, device
        )
        
        # Run optimization
        best_params, all_params, all_fitness = run_optimization(
            model, images, labels, rule_type, initial_params, args
        )
        
        # Apply constraints to optimized parameters
        constrained_best = constrain_parameters(best_params, rule_type)
        
        # Evaluate optimized relevance
        best_relevance, optimized_scores = evaluate_optimized_relevance(
            model, images, labels, rule_type, constrained_best, args
        )
        
        # Visualize comparison
        visualize_comparison(
            images, initial_relevance, best_relevance, initial_scores, optimized_scores,
            labels, args.dataset, rule_type, args, device
        )
        
        # Visualize evolution
        visualize_evolution(all_params, all_fitness, initial_params, rule_type, args)
        
        # Save parameters
        save_parameters(constrained_best, rule_type, args)
        
        # Calculate and display improvement
        avg_initial = sum(initial_scores) / len(initial_scores)
        avg_optimized = sum(optimized_scores) / len(optimized_scores)
        improvement = avg_optimized - avg_initial
        improvement_percent = (improvement / abs(avg_initial)) * 100 if avg_initial != 0 else float('inf')
        
        logger.info(f"Optimization results summary:")
        logger.info(f"  Initial average {args.criterion} score: {avg_initial:.4f}")
        logger.info(f"  Optimized average {args.criterion} score: {avg_optimized:.4f}")
        logger.info(f"  Absolute improvement: {improvement:.4f}")
        logger.info(f"  Relative improvement: {improvement_percent:.2f}%")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Optimization interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    sys.exit(main())
