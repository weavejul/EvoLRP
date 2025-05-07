import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import argparse
import json
import os
import sys
import logging
from pathlib import Path

# Project-specific imports
from lrp import UniformRuleLRP, UniformValueLRP, UniformLayerLRP
from datasets import get_mnist_data, get_imagenet_data
from models import load_model
from utils.visualization import plot_heatmap

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("LRP-Visual-Editor")

# Define constants
AVAILABLE_RULES = ['alpha_beta', 'epsilon', 'gamma']
AVAILABLE_DATASETS = ['mnist', 'imagenet']
AVAILABLE_DEVICES = ['cpu', 'cuda', 'mps']
AVAILABLE_LRP_TYPES = ['UniformRule', 'UniformValue', 'UniformLayer']

# Default values for rule parameters
DEFAULT_RULE_VALUES = {
    'alpha_beta': 2.0,  # Default alpha value (beta is alpha-1)
    'epsilon': 0.25,    # Default epsilon value
    'gamma': 0.25       # Default gamma value
}

class LRPVisualEditor:
    def __init__(self, args):
        """Initialize the LRP Visual Editor with configuration."""
        self.args = args
        self.device = self._setup_device()
        self.model, self.images, self.labels, self.class_names = self._load_data_and_model()
        
        # Get model predictions
        with torch.no_grad():
            self.outputs = self.model(self.images)
        
        # Calculate probabilities and sort by most probable
        self.probs = torch.softmax(self.outputs, dim=1)
        self.sorted_probs, self.sorted_indices = torch.sort(self.probs, dim=1, descending=True)
        
        # Select image to focus on
        self.selected_img_idx = 0
        
        # Load or initialize LRP parameters
        self.lrp_instance = self._setup_lrp_instance()
        
        # Current class RANK being viewed (index into sorted_indices)
        self.current_class_rank_idx = 0
        self.max_classes_to_compute = min(20, self.outputs.shape[1]) # Limit computation for averaging

        # Store relevance maps
        self.all_rank_relevance_maps = [] # Stores maps for top N ranks
        self.average_relevance_map = None # Average of the top N maps
        self.processed_relevance_map = None # Difference map (Selected - Average)
        
        # Initialize plot objects
        self.heatmap_im = None
        self.colorbar = None
        self.fixed_vmin = None # Store fixed color limits for the processed map
        self.fixed_vmax = None # Store fixed color limits for the processed map
        
        # Compute initial relevance maps (all ranks + average)
        self._update_all_relevance_maps()
        # Compute initial processed map for rank 0
        self._compute_processed_map()
        
        # Set up the interactive plot
        self._setup_interactive_plot()
    
    def _setup_device(self):
        """Set up the compute device."""
        if self.args.device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        else:
            device = self.args.device
            
        # Set GPU device if using CUDA
        if device == 'cuda':
            if self.args.gpu_id >= torch.cuda.device_count():
                logger.warning(f"GPU {self.args.gpu_id} not available, falling back to GPU 0")
                self.args.gpu_id = 0
            torch.cuda.set_device(self.args.gpu_id)
            logger.info(f"Using GPU {self.args.gpu_id}")
            
        logger.info(f"Using device: {device}")
        return torch.device(device)
    
    def _load_data_and_model(self):
        """Load dataset and model based on configuration."""
        try:
            # Load dataset
            if self.args.dataset == 'mnist':
                dataloader = get_mnist_data(batch_size=self.args.batch_size)
                class_names = [str(i) for i in range(10)]  # MNIST has 10 classes (0-9)
                logger.info(f"Loaded MNIST dataset with batch size {self.args.batch_size}")
                
                # Load MNIST model
                if self.args.model_path:
                    model = load_model(self.args.model_path)
                    logger.info(f"Loaded custom MNIST model from {self.args.model_path}")
                else:
                    # Correct the path if necessary, assuming it's relative to project root or similar
                    default_mnist_path = Path('models/mnist_cnn.pth') # Use Path for better handling
                    if not default_mnist_path.exists():
                         # Try relative to script location if not found directly
                         script_dir = Path(__file__).parent
                         default_mnist_path = script_dir.parent / 'models/mnist_cnn.pth' # Adjust relative path
                         if not default_mnist_path.exists():
                              logger.error("Default MNIST model not found at expected locations.")
                              sys.exit(1) # Exit if model cannot be found
                    
                    model = load_model(str(default_mnist_path))
                    logger.info(f"Loaded default MNIST model from {default_mnist_path}")
            else:  # ImageNet with VGG16
                # Configure ImageNet loader
                config = argparse.Namespace()
                config.input_dir = self.args.data_dir
                config.batch_size = self.args.batch_size
                config.resize = self.args.image_size
                config.subset_size = self.args.subset_size
                
                # Load class names
                class_names = []
                # First try loading from the same directory as this script
                script_dir = os.path.dirname(os.path.abspath(__file__))
                imagenet_classes_path = os.path.join(script_dir, 'imagenet_classes.txt')
                
                if os.path.exists(imagenet_classes_path):
                    logger.info(f"Loading ImageNet classes from {imagenet_classes_path}")
                    try:
                        with open(imagenet_classes_path, 'r') as f:
                            # Assuming it's a dictionary-like format
                            classes_dict = eval(f.read())
                            class_names = [classes_dict[i] for i in range(1000)]
                    except Exception as e:
                        logger.error(f"Error parsing imagenet_classes.txt: {e}")
                        # Fallback to numbered classes if file exists but can't be parsed
                        class_names = [f"Class {i}" for i in range(1000)]
                else:
                    # Fallback to numbered classes if file doesn't exist
                    logger.warning(f"ImageNet classes file not found at {imagenet_classes_path}")
                    class_names = [f"Class {i}" for i in range(1000)]
                
                try:
                    dataloader = get_imagenet_data(config)
                    logger.info(f"Loaded ImageNet dataset")
                except Exception as e:
                    logger.error(f"Failed to load ImageNet dataset: {e}")
                    logger.error(f"Please make sure the dataset is available at {self.args.data_dir}")
                    sys.exit(1)
                
                # Load VGG16 model
                if self.args.model_path:
                    logger.info(f"Loading custom model from {self.args.model_path}")
                    try:
                        model = torch.load(self.args.model_path)
                    except Exception as e:
                        logger.error(f"Failed to load custom model: {e}")
                        sys.exit(1)
                else:
                    logger.info("Loading pretrained VGG16 model")
                    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            
            # Move model to device and set to eval mode
            model = model.to(self.device)
            model.eval()
            
            # Get a batch of data
            try:
                images, labels = next(iter(dataloader))
                logger.info(f"Processed batch of {images.shape[0]} images")
            
                # Move data to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                return model, images, labels, class_names
                
            except StopIteration:
                logger.error("Failed to get data batch. Dataloader is empty.")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Error loading data and model: {e}")
            raise
    
    def _setup_lrp_instance(self):
        """Set up LRP instance based on the specified type and parameters."""
        try:
            lrp_type = self.args.lrp_type
            
            if self.args.params_file:
                # Load parameters from file
                logger.info(f"Loading parameters from {self.args.params_file}")
                try:
                    with open(self.args.params_file, 'r') as f:
                        params_data = json.load(f)
                    logger.info(f"Successfully loaded parameters structure: {type(params_data)}")
                except Exception as e:
                    logger.error(f"Failed to load or parse parameters file {self.args.params_file}: {e}")
                    raise  # Re-raise after logging

                # Check if the loaded data is a dictionary (standard format) or just a list (legacy/simple UniformRule format)
                if isinstance(params_data, dict):
                    # --- Standard Dictionary Format Handling --- 
                    # Determine LRP type from file if available, otherwise use command-line arg
                    file_lrp_type = params_data.get('optimization_type', lrp_type) # Check for 'optimization_type' key
                    if file_lrp_type != lrp_type:
                        logger.warning(f"LRP type mismatch: Command line '{lrp_type}', File '{file_lrp_type}'. Using type from file.")
                        lrp_type = file_lrp_type # Prioritize type from file if specified

                    logger.info(f"Setting up LRP instance of type: {lrp_type} based on dictionary structure")

                    # Create LRP instance based on loaded parameters and determined type
                    if lrp_type == 'UniformRule':
                        rule_type = params_data.get('rule_type', self.args.rule)
                        parameters_list = params_data.get('parameters')
                        if parameters_list is None:
                            logger.error("'parameters' key missing in JSON dictionary for UniformRuleLRP.")
                            raise ValueError("'parameters' key missing in JSON dictionary for UniformRuleLRP.")
                        parameters = torch.tensor(parameters_list, device=self.device)
                        logger.info(f"Instantiating UniformRuleLRP with rule '{rule_type}' and {len(parameters)} parameters.")
                        return UniformRuleLRP(self.model, rule_type=rule_type, rule_params=parameters, correct_class_only=False)
                    
                    elif lrp_type == 'UniformValue':
                        rule_types = params_data.get('rule_types')
                        rule_values = params_data.get('rule_values')
                        if rule_types is None or rule_values is None:
                            logger.error("Missing 'rule_types' or 'rule_values' in JSON dictionary for UniformValueLRP.")
                            raise ValueError("Missing required keys for UniformValueLRP in JSON dictionary.")
                        rule_values = {k: float(v) for k, v in rule_values.items()}
                        logger.info(f"Instantiating UniformValueLRP with {len(rule_types)} rule type assignments and values: {rule_values}")
                        return UniformValueLRP(self.model, rule_types=rule_types, rule_values=rule_values, correct_class_only=False)
                    
                    elif lrp_type == 'UniformLayer':
                        layer_rule_mapping = params_data.get('layer_rule_mapping')
                        layer_value_mapping = params_data.get('layer_value_mapping')
                        if layer_rule_mapping is None or layer_value_mapping is None:
                            logger.error("Missing 'layer_rule_mapping' or 'layer_value_mapping' in JSON dictionary for UniformLayerLRP.")
                            raise ValueError("Missing required keys for UniformLayerLRP in JSON dictionary.")
                        layer_value_mapping = {k: float(v) for k, v in layer_value_mapping.items()}
                        logger.info(f"Instantiating UniformLayerLRP with rule map: {layer_rule_mapping} and value map: {layer_value_mapping}")
                        return UniformLayerLRP(self.model, layer_rule_mapping=layer_rule_mapping, layer_value_mapping=layer_value_mapping, correct_class_only=False)
                    
                    else:
                        logger.error(f"Unsupported LRP type specified in parameters file or command line: {lrp_type}")
                        raise ValueError(f"Unsupported LRP type: {lrp_type}")

                elif isinstance(params_data, list):
                    # --- List Format Handling (Assume UniformRule parameters) ---
                    logger.info(f"Loaded parameters as a list. Assuming UniformRule parameters with rule type '{self.args.rule}' from command line.")
                    if lrp_type != 'UniformRule':
                         logger.warning(f"Parameter file contains a list, but command line LRP type is '{lrp_type}'. Forcing UniformRule.")
                         lrp_type = 'UniformRule' # Force UniformRule if params are just a list

                    parameters_list = params_data
                    rule_type = self.args.rule # Use rule from command line arg
                    parameters = torch.tensor(parameters_list, device=self.device)
                    logger.info(f"Instantiating UniformRuleLRP with rule '{rule_type}' and {len(parameters)} parameters from list.")
                    return UniformRuleLRP(self.model, rule_type=rule_type, rule_params=parameters, correct_class_only=False)
                
                else:
                     # Handle unexpected data type
                     logger.error(f"Loaded parameters file contains unexpected data type: {type(params_data)}. Expected dict or list.")
                     raise TypeError(f"Unexpected data type in parameters file: {type(params_data)}")

            else:
                # Create LRP instance with default parameters if no file specified
                logger.info(f"No params_file specified. Creating default LRP instance of type: {lrp_type}")
                if lrp_type == 'UniformRule':
                    # Count layers for parameter initialization
                    num_layers = 0
                    if hasattr(self.model, 'features'):
                        num_layers += len(list(self.model.features))
                    if hasattr(self.model, 'classifier'):
                        num_layers += len(list(self.model.classifier))
                    
                    # Initialize parameters with uniform value
                    parameters = torch.ones(num_layers, device=self.device) * self.args.init_value
                    
                    return UniformRuleLRP(
                        self.model, 
                        rule_type=self.args.rule, 
                        rule_params=parameters,
                        correct_class_only=False
                    )
                
                elif lrp_type == 'UniformValue':
                    # Count layers for rule type initialization
                    num_layers = 0
                    if hasattr(self.model, 'features'):
                        num_layers += len(list(self.model.features))
                    if hasattr(self.model, 'classifier'):
                        num_layers += len(list(self.model.classifier))
                    
                    # Initialize all layers with the same rule type
                    rule_types = [self.args.rule] * num_layers
                    
                    return UniformValueLRP(
                        self.model,
                        rule_types=rule_types,
                        rule_values=DEFAULT_RULE_VALUES, # Use the global defaults
                        correct_class_only=False
                    )
                
                elif lrp_type == 'UniformLayer':
                    # First identify all layer types in the model
                    layer_types = set()
                    if hasattr(self.model, 'features'):
                        for layer in self.model.features:
                            layer_types.add(layer.__class__.__name__)
                    if hasattr(self.model, 'classifier'):
                        for layer in self.model.classifier:
                            layer_types.add(layer.__class__.__name__)
                    
                    # Create mappings with appropriate rules for each layer type
                    layer_rule_mapping = {}
                    for layer_type in layer_types:
                        # Use Zero rule for non-important layers
                        if layer_type in ['Dropout', 'ReLU', 'MaxPool2d', 'BatchNorm2d']:
                            layer_rule_mapping[layer_type] = 'zero'
                        else:
                            layer_rule_mapping[layer_type] = self.args.rule
                    
                    # Use init_value for all layer types initially
                    layer_value_mapping = {layer_type: self.args.init_value for layer_type in layer_types}
                    
                    return UniformLayerLRP(
                        self.model,
                        layer_rule_mapping=layer_rule_mapping,
                        layer_value_mapping=layer_value_mapping,
                        correct_class_only=False
                    )
                
                else:
                    raise ValueError(f"Unsupported LRP type: {lrp_type}")
        
        except Exception as e:
            logger.error(f"Error setting up LRP instance: {e}")
            raise
    
    def _update_all_relevance_maps(self):
        """Compute relevance maps for the top N classes and their average."""
        try:
            logger.info(f"Calculating relevance maps for top {self.max_classes_to_compute} classes for image {self.selected_img_idx}...")
            
            # Get only the selected image
            selected_image = self.images[self.selected_img_idx].unsqueeze(0)
            
            calculated_maps = []
            for rank_idx in range(self.max_classes_to_compute):
                target_class = self.sorted_indices[self.selected_img_idx, rank_idx].item()
                target = torch.tensor([target_class], device=self.device)
                
                # Compute relevance map for this rank
                with torch.no_grad():
                    relevance_map = self.lrp_instance.compute_relevance(selected_image, target)
                    calculated_maps.append(relevance_map)
                
                logger.debug(f"  Rank {rank_idx} (Class {target_class}): Mean={relevance_map.mean():.4f}, Sum={relevance_map.sum():.4f}")

            # Store all maps
            self.all_rank_relevance_maps = calculated_maps
            
            # Calculate and store average map
            if self.all_rank_relevance_maps:
                stacked_maps = torch.stack(self.all_rank_relevance_maps)
                self.average_relevance_map = torch.mean(stacked_maps, dim=0)
                logger.info(f"Computed average relevance map. Mean={self.average_relevance_map.mean():.4f}, Sum={self.average_relevance_map.sum():.4f}")
            else:
                 self.average_relevance_map = None
                 logger.warning("No relevance maps were calculated to compute an average.")

        except Exception as e:
            logger.error(f"Error updating all relevance maps: {e}")
            # Reset stored maps on error to avoid using stale data
            self.all_rank_relevance_maps = []
            self.average_relevance_map = None
            self.processed_relevance_map = None
            raise

    def _compute_processed_map(self):
        """Computes the processed map by subtracting the average of highly correlated maps."""
        correlation_threshold = 0.8 # Threshold for considering maps correlated
        min_std_dev = 1e-9 # Minimum standard deviation to avoid division by zero in correlation

        if (self.current_class_rank_idx < 0 or 
            self.current_class_rank_idx >= len(self.all_rank_relevance_maps)):
            logger.warning("Cannot compute processed map: Selected rank index out of bounds.")
            self.processed_relevance_map = None
            return

        selected_map_tensor = self.all_rank_relevance_maps[self.current_class_rank_idx]
        # Flatten and ensure it's float64 for numerical stability in correlation calculation
        selected_map_flat = selected_map_tensor.flatten().cpu().numpy().astype(np.float64)
        std_selected = np.std(selected_map_flat)

        correlated_maps_tensors = []

        logger.debug(f"Calculating correlations for Rank {self.current_class_rank_idx}...")
        for i, other_map_tensor in enumerate(self.all_rank_relevance_maps):
            if i == self.current_class_rank_idx:
                continue # Skip self-correlation

            other_map_flat = other_map_tensor.flatten().cpu().numpy().astype(np.float64)
            std_other = np.std(other_map_flat)

            correlation = 0.0 # Default correlation
            # Calculate Pearson correlation only if both maps have non-zero variance
            if std_selected > min_std_dev and std_other > min_std_dev:
                 try:
                     # Calculate covariance matrix, take off-diagonal element
                     covariance_matrix = np.cov(selected_map_flat, other_map_flat)
                     # Check if covariance matrix computation was successful (can fail if input is constant)
                     if covariance_matrix is not None and covariance_matrix.shape == (2, 2):
                         covariance = covariance_matrix[0, 1]
                         correlation = covariance / (std_selected * std_other)
                     else:
                         logger.warning(f"Could not compute covariance for Rank {i}. Skipping correlation.")
                 except Exception as e:
                      logger.warning(f"Error computing correlation for Rank {i}: {e}. Skipping correlation.")
            
            logger.debug(f"  Correlation between Rank {self.current_class_rank_idx} and Rank {i}: {correlation:.4f}")

            if correlation >= correlation_threshold:
                correlated_maps_tensors.append(other_map_tensor)
                logger.debug(f"  Rank {i} map added (correlation >= {correlation_threshold})")

        if correlated_maps_tensors:
            # Compute average of the highly correlated maps
            stacked_correlated = torch.stack(correlated_maps_tensors)
            average_correlated_map = torch.mean(stacked_correlated, dim=0)
            # Compute difference
            self.processed_relevance_map = selected_map_tensor - average_correlated_map
            
            target_class = self.sorted_indices[self.selected_img_idx, self.current_class_rank_idx].item()
            logger.info(f"Computed CORRELATION-BASED processed map for Rank {self.current_class_rank_idx} (Class {target_class}) " +
                        f"(subtracted avg of {len(correlated_maps_tensors)} correlated maps). " +
                        f"Result Mean={self.processed_relevance_map.mean():.4f}, Sum={self.processed_relevance_map.sum():.4f}")
        else:
            # No highly correlated maps found, display the original selected map for now
            self.processed_relevance_map = selected_map_tensor
            target_class = self.sorted_indices[self.selected_img_idx, self.current_class_rank_idx].item()
            logger.warning(f"No maps found with correlation >= {correlation_threshold} for Rank {self.current_class_rank_idx} (Class {target_class}). Displaying ORIGINAL map.")
            # Optional fallback: Subtract overall average if available?
            # if self.average_relevance_map is not None:
            #     self.processed_relevance_map = selected_map_tensor - self.average_relevance_map
            #     logger.warning("Falling back to subtracting overall average map.")

    def _initial_plot_heatmap(self, relevance_map):
        """Plot the initial heatmap (using the processed map) and create the colorbar."""
        if relevance_map is None:
             logger.error("Initial relevance map is None, cannot plot heatmap.")
             # Create a dummy plot or return None? For now, let's create dummy data.
             heatmap = np.zeros((224, 224)) # Assuming ImageNet size, adjust if needed
        else:
            # Convert processed map to numpy array and handle shape
            heatmap = relevance_map.squeeze().cpu().numpy()
            
            # Handle multi-channel relevance maps (shouldn't happen for processed map, but keep for safety)
            if len(heatmap.shape) == 3 and heatmap.shape[0] == 3:
                heatmap = np.mean(heatmap, axis=0)

        # --- Apply Contrast Enhancement (Power Scaling) ---
        contrast_power = 2.0 # Power to raise magnitudes to (e.g., 2.0 for squaring)
        signed_heatmap = np.sign(heatmap)
        scaled_heatmap = signed_heatmap * (np.abs(heatmap) ** contrast_power)
        heatmap_to_plot = scaled_heatmap
        # ---------------------------------------------------
        
        cmap = 'bwr' # Good for differences around zero
        # Calculate limits based on the SCALED heatmap
        abs_scaled_heatmap = np.abs(heatmap_to_plot)
        vmax = np.max(abs_scaled_heatmap) if np.max(abs_scaled_heatmap) > 1e-9 else 1.0 # Avoid vmax=0
        vmin = -vmax
        
        # Plot the initial SCALED heatmap image
        im = self.ax[1].imshow(heatmap_to_plot, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax)
        
        # Create and store the colorbar
        self.colorbar = self.fig.colorbar(im, ax=self.ax[1], fraction=0.046, pad=0.04)

        # Store the initial vmin/vmax to keep the color scale consistent for the processed map
        self.fixed_vmin = vmin
        self.fixed_vmax = vmax
        logger.info(f"Set fixed color limits for processed map: vmin={self.fixed_vmin:.4f}, vmax={self.fixed_vmax:.4f}")
        
        # Set initial title
        self._set_heatmap_title() # Title should reflect the processed nature

        return im

    def _update_heatmap_plot(self, relevance_map):
        """Update the heatmap data (using the processed map) and color limits."""
        if relevance_map is None:
            logger.warning("Cannot update heatmap plot: relevance_map is None.")
            # Optionally clear the plot or show a placeholder
            heatmap = np.zeros_like(self.heatmap_im.get_array()) # Keep shape
        else:
            # Convert processed map to numpy array and handle shape
            heatmap = relevance_map.squeeze().cpu().numpy()
            
            # Handle multi-channel relevance maps
            if len(heatmap.shape) == 3 and heatmap.shape[0] == 3:
                heatmap = np.mean(heatmap, axis=0)

            # --- Apply Contrast Enhancement (Power Scaling) ---
            contrast_power = 2.0 # Power to raise magnitudes to (e.g., 2.0 for squaring)
            signed_heatmap = np.sign(heatmap)
            scaled_heatmap = signed_heatmap * (np.abs(heatmap) ** contrast_power)
            heatmap_to_plot = scaled_heatmap
            # ---------------------------------------------------

        # Update heatmap data using the fixed color limits
        self.heatmap_im.set_data(heatmap_to_plot) # Set the SCALED data
        self.heatmap_im.set_clim(vmin=self.fixed_vmin, vmax=self.fixed_vmax)
        
        # Update title
        self._set_heatmap_title() # Title should reflect the processed nature

    def _set_heatmap_title(self):
        """Sets the title for the heatmap axis."""
        class_idx = self.sorted_indices[self.selected_img_idx, self.current_class_rank_idx].item()
        # Ensure class_names are available and index is valid
        if self.class_names and 0 <= class_idx < len(self.class_names):
            class_name = self.class_names[class_idx]
        else:
            class_name = f"Index {class_idx}" # Fallback if class names missing or index out of bounds
        prob = self.sorted_probs[self.selected_img_idx, self.current_class_rank_idx].item()
        # Modify title to indicate it's a processed map
        self.ax[1].set_title(f"Processed LRP (Rank {self.current_class_rank_idx} - Avg)\nClass {class_idx} ({class_name}), Prob: {prob:.4f}")


    def _setup_interactive_plot(self):
        """Set up the interactive plot with widgets."""
        # Create figure
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))
        plt.subplots_adjust(bottom=0.25)
        
        # Plot the initial image
        if self.args.dataset == 'imagenet':
            # Denormalize ImageNet images for visualization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            img_to_show = self.images[self.selected_img_idx].cpu() * std.cpu() + mean.cpu()
            img_to_show = img_to_show.permute(1, 2, 0).numpy()  # Convert to HWC format
            # Clip values to [0, 1] just in case denormalization results in values slightly outside
            img_to_show = np.clip(img_to_show, 0, 1) 
            self.ax[0].imshow(img_to_show)
        else:
            # For MNIST grayscale images
            self.ax[0].imshow(self.images[self.selected_img_idx].squeeze().cpu().numpy(), cmap='gray')
        
        # Set titles
        self.ax[0].set_title(f"Input Image {self.selected_img_idx}")
        
        # Plot the initial *processed* relevance map and store the image object
        self.heatmap_im = self._initial_plot_heatmap(self.processed_relevance_map)
        
        # Create slider for class selection (rank)
        ax_class = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.class_slider = Slider(
            ax=ax_class, 
            label='Class Rank', 
            valmin=0, 
            valmax=self.max_classes_to_compute - 1, # Use the limit
            valinit=0,
            valstep=1
        )
        
        # Create slider for image selection
        ax_image = plt.axes([0.25, 0.05, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        self.image_slider = Slider(
            ax=ax_image, 
            label='Image Index', 
            valmin=0, 
            valmax=len(self.images)-1, 
            valinit=0,
            valstep=1
        )
        
        # Set up update functions for the sliders
        def update_class(val):
            """Callback for class rank slider change."""
            new_rank_idx = int(round(val)) # Ensure integer index
            logger.info(f"update_class called: Slider value {val}, New rank index {new_rank_idx}, Previous rank index {self.current_class_rank_idx}")
            if new_rank_idx != self.current_class_rank_idx:
                self.current_class_rank_idx = new_rank_idx
                self._compute_processed_map() # Compute difference map for the new rank
                self._update_plot() # Update the visualization with the new processed map
            else:
                logger.info("update_class: Rank index did not change, skipping update.")
        
        def update_image(val):
            """Callback for image index slider change."""
            new_img_idx = int(round(val))
            logger.info(f"update_image called: Slider value {val}, New image index {new_img_idx}, Previous image index {self.selected_img_idx}")
            if new_img_idx != self.selected_img_idx:
                self.selected_img_idx = new_img_idx

                # Recalculate ALL relevance maps and the average for the NEW image
                self._update_all_relevance_maps() 
                # Compute the processed map for the current rank on the new image's data
                self._compute_processed_map() 
                
                # Update image plot
                self.ax[0].clear()
                if self.args.dataset == 'imagenet':
                    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
                    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
                    img_to_show = self.images[self.selected_img_idx].cpu() * std.cpu() + mean.cpu()
                    img_to_show = img_to_show.permute(1, 2, 0).numpy()
                    img_to_show = np.clip(img_to_show, 0, 1) # Clip values
                    self.ax[0].imshow(img_to_show)
                else:
                    self.ax[0].imshow(self.images[self.selected_img_idx].squeeze().cpu().numpy(), cmap='gray')
                
                self.ax[0].set_title(f"Input Image {self.selected_img_idx}")
                
                # Update heatmap with the new processed map and redraw
                # We also need to reset the fixed color limits based on the new image's rank 0 processed map
                self._reset_color_limits_and_update_plot() 
            else:
                 logger.info("update_image: Image index did not change, skipping update.")

        # Register the update functions
        self.class_slider.on_changed(update_class)
        self.image_slider.on_changed(update_image)
        
        # Create a button to save the current visualization
        ax_save = plt.axes([0.8, 0.01, 0.1, 0.03])
        self.save_button = Button(ax_save, 'Save')
        
        def save_current_view(event):
            # Create output directory if it doesn't exist
            output_dir = Path('lrp_editor_output')
            output_dir.mkdir(exist_ok=True)
            
            # Generate a filename including rank
            class_idx = self.sorted_indices[self.selected_img_idx, self.current_class_rank_idx].item()
            filename = output_dir / f"img{self.selected_img_idx}_rank{self.current_class_rank_idx}_class{class_idx}_processed.png"
            
            # Save the figure
            self.fig.savefig(str(filename), dpi=300, bbox_inches='tight')
            logger.info(f"Saved processed visualization to {filename}")
        
        self.save_button.on_clicked(save_current_view)
    
    def _update_plot(self):
        """Update the heatmap plot with the current processed map and redraw."""
        logger.info(f"_update_plot: Updating heatmap using processed relevance map.")
        # Pass the processed map to the update function
        self._update_heatmap_plot(self.processed_relevance_map) 
        self.fig.canvas.draw_idle()
        logger.info("_update_plot: Canvas redraw requested.")

    def _reset_color_limits_and_update_plot(self):
        """Recalculate fixed color limits based on the current processed map and update plot."""
        if self.processed_relevance_map is None:
            logger.error("Cannot reset color limits: processed map is None.")
            return

        # Recalculate vmin/vmax from the current processed map (e.g., for rank 0 of new image)
        heatmap_np = self.processed_relevance_map.squeeze().cpu().numpy()
        if len(heatmap_np.shape) == 3 and heatmap_np.shape[0] == 3:
             heatmap_np = np.mean(heatmap_np, axis=0)
        
        # --- Apply Contrast Enhancement (Power Scaling) before calculating new limits ---
        contrast_power = 2.0
        signed_heatmap_np = np.sign(heatmap_np)
        scaled_heatmap_np = signed_heatmap_np * (np.abs(heatmap_np) ** contrast_power)
        # Calculate limits based on the SCALED heatmap
        abs_scaled_heatmap = np.abs(scaled_heatmap_np)
        # -----------------------------------------------------------------------------
        
        new_vmax = np.max(abs_scaled_heatmap) if np.max(abs_scaled_heatmap) > 1e-9 else 1.0
        new_vmin = -new_vmax

        # Update stored limits
        self.fixed_vmin = new_vmin
        self.fixed_vmax = new_vmax
        logger.info(f"Reset fixed color limits based on new image/rank 0: vmin={self.fixed_vmin:.4f}, vmax={self.fixed_vmax:.4f}")
        
        # Now update the plot using these new limits
        self._update_plot()

    def run(self):
        """Run the visual editor."""
        # Save the initial plot before showing
        output_dir = Path('lrp_editor_output')
        output_dir.mkdir(exist_ok=True)
        # Update filename to reflect processed nature
        filename = output_dir / f"img{self.selected_img_idx}_rank{self.current_class_rank_idx}_processed_initial.png"
        self.fig.savefig(str(filename), dpi=300, bbox_inches='tight')
        logger.info(f"Saved initial processed visualization to {filename}")
        
        plt.show()


def parse_arguments():
    """Parse command line arguments for the LRP Visual Editor."""
    parser = argparse.ArgumentParser(
        description="LRP Visual Editor - Interactive visualization for Layer-wise Relevance Propagation",
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
    
    # LRP configuration
    parser.add_argument('--lrp_type', type=str, default='UniformRule', choices=AVAILABLE_LRP_TYPES,
                        help=f'Type of LRP to use: {", ".join(AVAILABLE_LRP_TYPES)}')
    parser.add_argument('--rule', type=str, default='alpha_beta', choices=AVAILABLE_RULES,
                        help=f'LRP rule type: {", ".join(AVAILABLE_RULES)}')
    parser.add_argument('--init_value', type=float, default=2.0,
                        help='Initial value for rule parameters (used if no params_file)')
    parser.add_argument('--params_file', type=str, default=None,
                        help='Path to saved LRP parameters JSON file (from main.py)')
    
    # Hardware configuration
    parser.add_argument('--device', type=str, default=None, choices=AVAILABLE_DEVICES,
                        help=f'Device to use: {", ".join(AVAILABLE_DEVICES)}. If None, will use CUDA if available.')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use when device is cuda (e.g. 0, 1, etc)')
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set default data directory for ImageNet
    if args.data_dir is None and args.dataset == 'imagenet':
        # Assume data directory relative to project root
        project_root = Path(__file__).parent.parent.parent 
        args.data_dir = str(project_root / 'data' / 'ImageNet')
        logger.info(f"Defaulting ImageNet data directory to: {args.data_dir}")

    
    return args


def main():
    """Main function for LRP Visual Editor."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        logger.info(f"Starting LRP Visual Editor for {args.dataset} dataset")
        
        # Create and run the visual editor
        editor = LRPVisualEditor(args)
        editor.run()
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Visual editor terminated by user")
        return 1
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main()) 