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
from optim import CMAESOptimizer, CMAMOESOptimizer
from evaluation import pixel_flipping, average_sensitivity, sparseness, create_fitness_function, faithfulness_correlation
from datasets import get_mnist_data, get_imagenet_data
from models import load_model
from lrp import UniformRuleLRP, UniformValueLRP
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

# Default values for uniform layer rules
DEFAULT_RULE_VALUES = {
    'alpha_beta': 2.0,  # Default alpha value (beta is alpha-1)
    'epsilon': 0.25,     # Default epsilon value
    'gamma': 0.25       # Default gamma value
}

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
    
    # Optimization type configuration
    parser.add_argument('--optimization_type', type=str, default='UniformRule', choices=AVAILABLE_OPTIMIZATION_TYPES,
                        help=f'Type of optimization to perform: {", ".join(AVAILABLE_OPTIMIZATION_TYPES)}')
    parser.add_argument('--correct_class_only', action='store_true',
                        help='Calculate relevance only for the correct class instead of all logits')

        
    # LRP uniform rule configuration
    parser.add_argument('--rule', type=str, default='alpha_beta', choices=AVAILABLE_RULES,
                        help=f'LRP rule type: {", ".join(AVAILABLE_RULES)}')
    parser.add_argument('--init_value', type=float, default=0.25,
                        help='Initial value for all rule parameters')

    # LRP uniform value base values
    parser.add_argument('--alpha_beta_base', type=float, default=DEFAULT_RULE_VALUES['alpha_beta'],
                        help='Base value for alpha-beta rule (alpha value, beta is alpha-1)')
    parser.add_argument('--epsilon_base', type=float, default=DEFAULT_RULE_VALUES['epsilon'],
                        help='Base value for epsilon rule')
    parser.add_argument('--gamma_base', type=float, default=DEFAULT_RULE_VALUES['gamma'],
                        help='Base value for gamma rule')
    
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
    parser.add_argument('--mean_weight', type=float, default=0.8,
                        help='Weight for mean score in fitness function')
    parser.add_argument('--min_weight', type=float, default=0.2,
                        help='Weight for minimum score in fitness function')
    parser.add_argument('--multi_opt', action='store_true',
                        help='Use multi-objective optimization')
    
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
    
    # Update global rule values based on command line arguments
    global RULE_VALUES
    RULE_VALUES = {
        'alpha_beta': args.alpha_beta_base,
        'epsilon': args.epsilon_base,
        'gamma': args.gamma_base
    }
    
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
    optimization_type = args.optimization_type
    
    if optimization_type == 'UniformRule':
        # For UniformRuleLRP: every layer uses the same rule type with different parameters
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
            
        return initial_params, rule_type, None
        
    elif optimization_type == 'UniformValue':
        # For UniformValueLRP: layers can have different rule types, but same rule type shares parameter value
        # Start with all layers using the default rule type
        initial_rule_types = [rule_type] * num_layers
        
        logger.info(f"Initialized for UniformValue optimization:")
        logger.info(f"  Initial rule types: all layers set to {rule_type}")
        logger.info(f"  Rule values: {RULE_VALUES}")
        
        return initial_rule_types, rule_type, RULE_VALUES
    
    else:
        logger.error(f"Unsupported optimization type: {optimization_type}")
        sys.exit(1)
        
def evaluate_initial_relevance(model, images, labels, rule_type, initial_params, args):
    """Compute and evaluate initial relevance maps."""
    try:
        optimization_type = args.optimization_type
        
        # Create initial LRP instance based on optimization type
        if optimization_type == 'UniformRule':
            # For UniformRuleLRP, initial_params is a tensor of parameter values
            lrp_initial = UniformRuleLRP(model, rule_type=rule_type, rule_params=initial_params, 
                                       correct_class_only=args.correct_class_only)
        elif optimization_type == 'UniformValue':
            # For UniformValueLRP, initial_params is a list of rule types
            
            lrp_initial = UniformValueLRP(model, rule_types=initial_params, rule_values=RULE_VALUES,
                                        correct_class_only=args.correct_class_only)
        else:
            logger.error(f"Unsupported optimization type: {optimization_type}")
            raise ValueError(f"Unsupported optimization type: {optimization_type}")
        
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
                    num_steps=args.num_steps, baseline=args.baseline, attr_model=lrp_initial, method='lrp'
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
        # Create fitness function with proper parameters
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
            correct_class_only=args.correct_class_only,
            optimization_type=args.optimization_type,
        )
        
        num_params = None
        # Determine number of parameters to optimize
        if args.optimization_type == 'UniformRule':
            # For UniformRuleLRP: optimize one parameter per layer
            num_params = len(initial_params)
        elif args.optimization_type == 'UniformValue':
            # For UniformValueLRP: optimize rule type selection for each layer
            num_params = len(initial_params)
        else:
            raise ValueError(f"Unsupported optimization type: {args.optimization_type}")
        
        # Configure CMA-ES optimizer
        optimizer = CMAESOptimizer(
            num_params=num_params, 
            init_sigma=args.init_sigma,
            population_size=args.population_size,
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
            if args.criterion == 'average_sensitivity':
                best_fitness = optimizer.es.result.fbest  # es.result not implemented
            else:
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

def run_optimization_multi(model, images, labels, rule_type, initial_params, args):
    """Run optimization process for LRP parameters."""
    try:
        # Create fitness function with proper parameters
        weights = (args.mean_weight, args.min_weight)
        fitness_function1 = create_fitness_function(
            model=model,
            images=images,
            labels=labels,
            criterion="sparseness",
            rule_type=rule_type,
            weights=weights,
            baseline=args.baseline,
            num_steps=args.num_steps,
            correct_class_only=args.correct_class_only,
            optimization_type=args.optimization_type,
        )
        
        fitness_function2 = create_fitness_function(
            model=model,
            images=images,
            labels=labels,
            criterion="pixel_flipping",
            rule_type=rule_type,
            weights=weights,
            baseline=args.baseline,
            num_steps=args.num_steps,
            correct_class_only=args.correct_class_only,
            optimization_type=args.optimization_type,
        )

        num_params = None
        # Determine number of parameters to optimize
        if args.optimization_type == 'UniformRule':
            # For UniformRuleLRP: optimize one parameter per layer
            num_params = len(initial_params)
        elif args.optimization_type == 'UniformValue':
            # For UniformValueLRP: optimize rule type selection for each layer
            num_params = len(initial_params)
        else:
            raise ValueError(f"Unsupported optimization type: {args.optimization_type}")
        
        # Configure CMA-ES optimizer
        optimizer = CMAMOESOptimizer(
            num_params=num_params, 
            init_sigma=args.init_sigma,
            population_size=args.population_size,
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
            optimizer.step(fitness_function1, fitness_function2)
            
            # Print status periodically
            if gen % 5 == 0 or gen == args.generations - 1:
                logger.info(f"Generation {gen+1}/{args.generations}")
                optimizer.es.disp()
                
            # Track best parameters and fitness
            best_params = optimizer.best_solution()
            if args.criterion == 'average_sensitivity':
                best_fitness = optimizer.es.best_hypervolume_pareto_front  # Negate back to positive
            else:
                best_fitness = -optimizer.es.best_hypervolume_pareto_front  #not implemented result
            all_params.append(best_params)
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

def constrain_parameters(best_params, rule_type, args):
    """Apply constraints to best parameters based on rule type."""
    
    if args.optimization_type == 'UniformRule':
        # Original behavior for UniformRuleLRP
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
                constrained_best.append(max(0.1, min(3.0, abs(p))))
                    
            constrained_best = torch.tensor(constrained_best)
            logger.info(f"Constrained gamma parameters: {constrained_best.tolist()}")
        
        return constrained_best
        
    elif args.optimization_type == 'UniformValue':
        # For UniformValueLRP, best_params determines the rule type for each layer
        rule_mapping = {
            0: 'alpha_beta',
            1: 'epsilon',
            2: 'gamma'
        }
        
        # Convert continuous parameters to rule type assignments
        rule_types = []
        for i, p in enumerate(best_params):
            # Map the continuous parameter to one of the three rule types
            rule_index = int(abs(p) % 3)
            rule_types.append(rule_mapping[rule_index])
        
        # Count frequency of each rule type
        rule_counts = {rule: rule_types.count(rule) for rule in set(rule_types)}
        
        logger.info(f"Optimized rule type assignments:")
        logger.info(f"  Rule counts: {rule_counts}")
        logger.info(f"  Rule assignments: {rule_types}")
        
        # Return both rule types and rule values
        return {
            'rule_types': rule_types,
            'rule_values': RULE_VALUES
        }
    
    else:
        raise ValueError(f"Unsupported optimization type: {args.optimization_type}")

def evaluate_optimized_relevance(model, images, labels, rule_type, constrained_best, args):
    """Compute and evaluate optimized relevance maps."""
    try:
        # Create LRP with optimized parameters based on optimization type
        if args.optimization_type == 'UniformRule':
            # For UniformRuleLRP, constrained_best is a tensor of parameters
            lrp_best = UniformRuleLRP(model, rule_type=rule_type, rule_params=constrained_best, 
                                  correct_class_only=args.correct_class_only)
        elif args.optimization_type == 'UniformValue':
            # For UniformValueLRP, constrained_best is a dict with rule_types and rule_values
            lrp_best = UniformValueLRP(model, rule_types=constrained_best['rule_types'], 
                                    rule_values=constrained_best['rule_values'],
                                    correct_class_only=args.correct_class_only)
        else:
            raise ValueError(f"Unsupported optimization type: {args.optimization_type}")
        
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
                    num_steps=args.num_steps, baseline=args.baseline, attr_model=lrp_best
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
    """Save optimized parameters to a JSON file."""
    if not args.save_params:
        logger.info("Parameter saving is disabled. Use --save_params to enable.")
        return
        
    try:
        # Create output directory if it doesn't exist
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create output filename with appropriate prefix based on optimization type
        prefix = "UR_" if args.optimization_type == 'UniformRule' else "UL_"
        
        # For UniformRule, use different filename based on rule type
        if args.optimization_type == 'UniformRule':
            if rule_type == 'alpha_beta':
                suffix = "alphas"
            elif rule_type == 'epsilon':
                suffix = "epsilons"
            elif rule_type == 'gamma':
                suffix = "gammas"
            else:
                suffix = "params"
                
            timestamp = Path(args.output_dir) / f"{prefix}{suffix}_{args.criterion}_{args.dataset}.json"
            
            # Prepare data to save
            params_data = {
                "rule_type": rule_type,
                "criterion": args.criterion,
                "dataset": args.dataset,
                "correct_class_only": args.correct_class_only,
                "parameters": constrained_best.tolist()
            }
        else:  # UniformValue
            timestamp = Path(args.output_dir) / f"{prefix}rules_{args.criterion}_{args.dataset}.json"
            
            # Prepare data to save for UniformValue
            params_data = {
                "optimization_type": args.optimization_type,
                "criterion": args.criterion,
                "dataset": args.dataset,
                "correct_class_only": args.correct_class_only,
                "rule_types": constrained_best['rule_types'],
                "rule_values": {k: float(v) for k, v in constrained_best['rule_values'].items()}
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
        initial_rule_types, rule_type, rule_values = initialize_parameters(args, num_layers)
        
        # Evaluate initial relevance
        initial_relevance, initial_scores = evaluate_initial_relevance(
            model, images, labels, rule_type, initial_rule_types, args
        )
        
        # Visualize initial results
        visualize_initial_results(
            images, initial_relevance, initial_scores, labels, args.dataset, rule_type, args, device
        )
        
        # Run optimization
        if args.multi_opt:
            best_params, all_params, all_fitness = run_optimization_multi(
                model, images, labels, rule_type, initial_rule_types, args
            )
        else:
            best_params, all_params, all_fitness = run_optimization(
                model, images, labels, rule_type, initial_rule_types, args
            )
        
        # Apply constraints to optimized parameters
        constrained_best = constrain_parameters(best_params, rule_type, args)
        
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
        visualize_evolution(all_params, all_fitness, initial_rule_types, rule_type, args)
        
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
