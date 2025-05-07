import torch
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import os
import sys
import logging
from pathlib import Path

torch.manual_seed(42)

# Project-specific imports
# Assuming these modules are accessible from the script's location or PYTHONPATH is set
try:
    from lrp import UniformRuleLRP # Focus on UniformRule based on request
    from datasets import get_mnist_data, get_imagenet_data
    from models import load_model
    # Assuming visualization utils are available
    # from utils.visualization import plot_heatmap 
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print("Please ensure the script is run from the correct directory or PYTHONPATH is set.")
    sys.exit(1)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Combined-LRP-Visualizer")

# Define constants
AVAILABLE_DATASETS = ['mnist', 'imagenet']
AVAILABLE_DEVICES = ['cpu', 'cuda', 'mps']
# Add rule types that might be loaded from files
AVAILABLE_RULES = ['alpha_beta', 'epsilon', 'gamma'] 

class CombinedLRPVisualizer:
    """
    Interactively visualizes and combines relevance maps from multiple LRP parameter sets.
    """
    def __init__(self, args):
        """Initialize the visualizer with configuration."""
        self.args = args
        self.device = self._setup_device()
        self.model, self.images, self.labels, self.class_names = self._load_data_and_model()
        self.num_classes = self.model(self.images[0].unsqueeze(0)).shape[1] # Get num classes from model output
        
        # Check starting image index
        if self.args.img_idx >= len(self.images):
            logger.error(f"Starting image index {self.args.img_idx} out of bounds (0-{len(self.images)-1}). Defaulting to 0.")
            self.args.img_idx = 0
        self.current_img_idx = self.args.img_idx
        
        # Variables to store current image predictions
        self.current_selected_img = None
        self.current_probs = None
        self.current_sorted_probs = None
        self.current_sorted_indices = None
        self.current_target_class_idx = None # Set during interaction
        self.current_target_class_tensor = None
        self.current_target_class_prob = None

        # Store weights
        self.weights = {
            'sensitivity': self.args.weight_sensitivity,
            'sparseness': self.args.weight_sparseness,
            'faithfulness': self.args.weight_faithfulness
        }
        logger.info(f"Using weights: {self.weights}")

        # Load LRP instances (Load all specified, regardless of weight if diff mode)
        self.lrp_instances = {}
        self.param_files = {
            'sensitivity': self.args.params_sensitivity,
            'sparseness': self.args.params_sparseness,
            'faithfulness': self.args.params_faithfulness
        }
        for name, filepath in self.param_files.items():
            if filepath:
                instance = self._load_lrp_instance(filepath)
                if instance:
                    self.lrp_instances[name] = instance
                else:
                     logger.warning(f"Could not load LRP instance for '{name}' from {filepath}")
            else:
                logger.info(f"No parameter file specified for '{name}', skipping (required for combination).")
                
        if not self.lrp_instances:
             logger.error("No LRP instances were successfully loaded. Exiting.")
             sys.exit(1)
             
        # Filter instances based on weight for combination purposes
        self.active_lrp_instances = {
            name: inst for name, inst in self.lrp_instances.items() 
            if self.weights.get(name, 0) != 0
        }
        if not self.active_lrp_instances:
             logger.warning("All loaded LRP instances have zero weight. Combination map will not be generated.")

        # To store computed relevance maps for the selected class
        self.relevance_maps = {name: {} for name in self.lrp_instances}
        self.combined_relevance_map = None 
        
        # Matplotlib figure management for interactive mode
        self.fig = None
        self.axes = None

    def _setup_device(self):
        """Set up the compute device."""
        if self.args.device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                 # Check for MPS availability for MacOS
                 # Note: MPS support might vary based on PyTorch version and hardware
                 try:
                      torch.mps.is_available()
                      device = 'mps'
                 except AttributeError:
                      device = 'cpu' # Fallback if MPS is not recognized
            else:
                device = 'cpu'
        else:
            device = self.args.device
            
        if device == 'cuda':
            if self.args.gpu_id >= torch.cuda.device_count():
                logger.warning(f"GPU {self.args.gpu_id} not available, falling back to GPU 0")
                self.args.gpu_id = 0
            torch.cuda.set_device(self.args.gpu_id)
            logger.info(f"Using CUDA GPU {self.args.gpu_id}")
        elif device == 'mps':
             logger.info("Using Metal Performance Shaders (MPS) device on MacOS")
             # Additional MPS setup if needed could go here
        else:
             logger.info("Using CPU device")
             
        return torch.device(device)

    def _load_data_and_model(self):
        """Load dataset and model based on configuration."""
        # This function is adapted from lrp_visual_editor.py
        try:
            # Load dataset
            if self.args.dataset == 'mnist':
                dataloader = get_mnist_data(batch_size=self.args.batch_size)
                class_names = [str(i) for i in range(10)]
                logger.info(f"Loaded MNIST dataset with batch size {self.args.batch_size}")
                
                # Load MNIST model
                if self.args.model_path:
                    model = load_model(self.args.model_path)
                    logger.info(f"Loaded custom MNIST model from {self.args.model_path}")
                else:
                    default_mnist_path = Path('models/mnist_cnn.pth')
                    if not default_mnist_path.exists():
                         script_dir = Path(__file__).parent
                         default_mnist_path = script_dir.parent / 'models/mnist_cnn.pth'
                         if not default_mnist_path.exists():
                              logger.error("Default MNIST model not found at expected locations.")
                              sys.exit(1)
                    model = load_model(str(default_mnist_path))
                    logger.info(f"Loaded default MNIST model from {default_mnist_path}")

            elif self.args.dataset == 'imagenet':
                config = argparse.Namespace()
                config.input_dir = self.args.data_dir
                config.batch_size = self.args.batch_size # Use batch size for loading
                config.resize = self.args.image_size
                config.subset_size = self.args.subset_size
                
                class_names = []
                # Adjusted path finding for imagenet_classes.txt
                script_dir = Path(__file__).resolve().parent
                imagenet_classes_path = script_dir / 'imagenet_classes.txt'
                
                if imagenet_classes_path.exists():
                    logger.info(f"Loading ImageNet classes from {imagenet_classes_path}")
                    try:
                        with open(imagenet_classes_path, 'r') as f:
                            content = f.read()
                            if content.strip().startswith('{') and content.strip().endswith('}'):
                                classes_dict = eval(content) 
                                if isinstance(classes_dict, dict) and all(isinstance(k, int) for k in classes_dict.keys()):
                                     class_names = [classes_dict.get(i, f'Class {i}') for i in range(1000)]
                                else:
                                     raise ValueError("Parsed content is not a dictionary with integer keys.")
                            else:
                                raise ValueError("File content does not appear to be a Python dictionary literal.")
                    except Exception as e:
                        logger.error(f"Error parsing imagenet_classes.txt: {e}. Falling back to numbered classes.")
                        class_names = [f"Class {i}" for i in range(1000)]
                else:
                    logger.warning(f"ImageNet classes file not found at {imagenet_classes_path}. Using numbered classes.")
                    class_names = [f"Class {i}" for i in range(1000)]
                
                try:
                    # Load the whole batch specified by batch_size
                    dataloader = get_imagenet_data(config)
                    logger.info(f"Loading ImageNet data with batch size {self.args.batch_size}...")
                    all_images, all_labels = [], []
                    count = 0
                    for imgs, labs in dataloader:
                        all_images.append(imgs)
                        all_labels.append(labs)
                        count += len(imgs)
                        if count >= self.args.batch_size: # Stop once we have enough
                            break 
                    if not all_images:
                         raise StopIteration("Dataloader yielded no batches.")
                    images = torch.cat(all_images, dim=0)[:self.args.batch_size] # Concatenate and trim to exact batch size
                    labels = torch.cat(all_labels, dim=0)[:self.args.batch_size]
                    logger.info(f"Loaded {len(images)} images for interaction.")
                        
                except Exception as e:
                    logger.error(f"Failed to load ImageNet dataset: {e}")
                    logger.error(f"Please make sure the dataset is available at {self.args.data_dir}")
                    sys.exit(1)
                
                if self.args.model_path:
                    logger.info(f"Loading custom model from {self.args.model_path}")
                    try:
                        model = load_model(self.args.model_path)
                    except Exception as e:
                        logger.error(f"Failed to load custom model: {e}")
                        sys.exit(1)
                else:
                    logger.info("Loading pretrained VGG16 model")
                    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            else:
                 logger.error(f"Unsupported dataset: {self.args.dataset}")
                 sys.exit(1)

            model = model.to(self.device)
            model.eval()
            
            # Return the full batch of images/labels loaded
            images = images.to(self.device)
            labels = labels.to(self.device)
            return model, images, labels, class_names
                
        except Exception as e:
            logger.error(f"Error loading data and model: {e}", exc_info=True)
            raise

    def _load_lrp_instance(self, params_file_path):
        """Loads LRP parameters and creates a UniformRuleLRP instance."""
        try:
            filepath = Path(params_file_path)
            if not filepath.exists():
                 logger.error(f"Parameter file not found: {filepath}")
                 return None
                 
            logger.info(f"Loading parameters from {filepath}")
            with open(filepath, 'r') as f:
                params_data = json.load(f)
            logger.info(f"Successfully loaded parameters structure: {type(params_data)}")

            # Determine LRP type and rule type from file or infer
            lrp_type = 'UniformRule' # Explicitly set based on user request focus
            rule_type = None # Try to get from dict, otherwise infer/assume

            if isinstance(params_data, dict):
                # Newer dictionary format
                file_lrp_type = params_data.get('optimization_type')
                if file_lrp_type and file_lrp_type != lrp_type:
                     logger.warning(f"File indicates LRP type '{file_lrp_type}', but using 'UniformRule' as requested.")
                     # Potentially check if file_lrp_type is compatible or error out

                rule_type = params_data.get('rule_type') # Get rule type if specified
                parameters_list = params_data.get('parameters')
                if parameters_list is None:
                    logger.error(f"'parameters' key missing in JSON dict: {filepath}")
                    return None
                parameters = torch.tensor(parameters_list, device=self.device)

            elif isinstance(params_data, list):
                # Older list format (assume UniformRule parameters)
                logger.info("Loaded parameters as a list. Assuming UniformRule.")
                parameters_list = params_data
                parameters = torch.tensor(parameters_list, device=self.device)
                # Cannot infer rule type from list, need to rely on filename or assume
                # Let's try to infer from filename (e.g., 'opt_alpha_beta_...')
                filename = filepath.stem # e.g., opt_alpha_beta_sparseness
                for available_rule in AVAILABLE_RULES:
                    if available_rule in filename:
                        rule_type = available_rule
                        logger.info(f"Inferred rule type '{rule_type}' from filename.")
                        break
                if rule_type is None:
                    logger.warning(f"Could not infer rule type from filename {filepath.name}. Assuming 'alpha_beta'.")
                    rule_type = 'alpha_beta' # Default assumption

            else:
                 logger.error(f"Unsupported data type in parameters file: {type(params_data)} in {filepath}")
                 return None

            if rule_type is None:
                 # If still None after checks (e.g., dict format missing 'rule_type')
                 logger.warning(f"Rule type not specified in {filepath}. Assuming 'alpha_beta'.")
                 rule_type = 'alpha_beta' # Default assumption
            elif rule_type not in AVAILABLE_RULES:
                 logger.error(f"Unsupported rule type '{rule_type}' found in/inferred for {filepath}. Supported: {AVAILABLE_RULES}")
                 return None

            logger.info(f"Instantiating UniformRuleLRP with rule '{rule_type}' and {len(parameters)} parameters.")
            # Set correct_class_only=False based on lrp_visual_editor.py
            return UniformRuleLRP(self.model, rule_type=rule_type, rule_params=parameters, correct_class_only=False)

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {params_file_path}")
            return None
        except Exception as e:
            logger.error(f"Error setting up LRP instance from {params_file_path}: {e}", exc_info=True)
            return None
            
    def _predict_and_show_top(self):
        """Predicts classes for the current image and displays top N."""
        self.current_selected_img = self.images[self.current_img_idx].unsqueeze(0)
        logger.info(f"\n--- Analyzing Image Index: {self.current_img_idx} ---")
        with torch.no_grad():
            outputs = self.model(self.current_selected_img)
        self.current_probs = torch.softmax(outputs, dim=1)
        self.current_sorted_probs, self.current_sorted_indices = torch.sort(self.current_probs, dim=1, descending=True)

        print(f"Top {self.args.top_n_display} predictions for image {self.current_img_idx}:")
        for i in range(min(self.args.top_n_display, self.num_classes)):
            idx = self.current_sorted_indices[0, i].item()
            prob = self.current_sorted_probs[0, i].item()
            class_name = "N/A"
            if self.class_names and 0 <= idx < len(self.class_names):
                full_name = self.class_names[idx]
                class_name = full_name.split(',')[0] # Short name
            else:
                 class_name = f"Index {idx}"
            print(f"  Rank {i}: Idx={idx:<5} Prob={prob:.4f}  Name={class_name}")
        print("-"*20)
        # Return the set of valid indices shown to the user
        return set(self.current_sorted_indices[0,:self.args.top_n_display].cpu().numpy())


    def compute_relevance_maps(self):
        """Computes relevance maps based on the selected mode (standard or difference)."""
        if self.current_target_class_idx is None:
             logger.error("Cannot compute relevance: Target class index not set.")
             return
             
        # Reset maps for the new computation
        self.relevance_maps = {name: {} for name in self.lrp_instances}
        self.combined_relevance_map = None
        self.current_target_class_tensor = torch.tensor([self.current_target_class_idx], device=self.device)
        self.current_target_class_prob = self.current_probs[0, self.current_target_class_idx].item()
        
        if self.args.show_difference:
            logger.info(f"Computing relevance maps in DIFFERENCE mode (Target: {self.current_target_class_idx} vs Avg Top {self.args.diff_top_n} Others)...")
            self._compute_difference_maps()
        else:
            logger.info(f"Computing relevance maps in STANDARD mode (Target: {self.current_target_class_idx})...")
            self._compute_target_maps()

    def _compute_target_maps(self):
        """Computes relevance for the target class only and combines them."""
        computed_maps_for_combination = []
        # Use only active instances for combination
        for name, lrp_instance in self.active_lrp_instances.items():
            weight = self.weights.get(name, 1.0) # Should not be 0 here based on active_lrp_instances filter
            logger.info(f"  Computing relevance for '{name}' [Target Class] (Weight: {weight:.2f})...")
            try:
                with torch.no_grad():
                    # Use current_selected_img and current_target_class_tensor
                    r_target = lrp_instance.compute_relevance(self.current_selected_img, self.current_target_class_tensor)
                self.relevance_maps[name]['target'] = r_target
                computed_maps_for_combination.append(r_target * weight)
                logger.info(f"    Done for '{name}'. Map shape: {r_target.shape}, Sum: {r_target.sum():.4f}")
            except Exception as e:
                 logger.error(f"    Error computing target relevance for '{name}': {e}", exc_info=True)
        
        # Combine weighted target maps
        self.combined_relevance_map = self._combine_maps(computed_maps_for_combination, "target")
            
    def _compute_difference_maps(self):
        """Computes target, average-other, and difference maps, then combines difference maps."""
        computed_diff_maps_for_combination = []

        # Identify top N other classes FOR THE CURRENT IMAGE
        top_indices = self.current_sorted_indices[0].cpu().numpy()
        other_class_indices = [idx for idx in top_indices if idx != self.current_target_class_idx][:self.args.diff_top_n]
        
        if not other_class_indices:
            logger.warning("Could not find any other classes for difference calculation for this target. Skipping difference.")
            self.relevance_maps = {name: {} for name in self.lrp_instances} # Clear maps
            self.combined_relevance_map = None
            return # Go back to prompt
        
        logger.info(f"  Using Top-{len(other_class_indices)} other classes for averaging: {other_class_indices}")

        # Iterate through ALL loaded instances to compute necessary maps
        for name, lrp_instance in self.lrp_instances.items():
            logger.info(f"  Processing '{name}'...")
            weight = self.weights.get(name, 1.0) # Get weight for potential combination
            r_target, r_avg_others, r_diff = None, None, None
            
            # 1. Compute Target Relevance
            try:
                logger.info(f"    Computing relevance for Target Class ({self.current_target_class_idx})...")
                with torch.no_grad():
                    r_target = lrp_instance.compute_relevance(self.current_selected_img, self.current_target_class_tensor)
                self.relevance_maps[name]['target'] = r_target
                logger.info(f"      Target Map Sum: {r_target.sum():.4f}")
            except Exception as e:
                logger.error(f"    Error computing target relevance for '{name}': {e}", exc_info=True)
                self.relevance_maps[name]['target'] = None

            # 2. Compute Average Relevance of Other Classes
            other_maps = []
            for other_idx in other_class_indices:
                try:
                    logger.debug(f"    Computing relevance for Other Class ({other_idx})...")
                    other_target_tensor = torch.tensor([other_idx], device=self.device)
                    with torch.no_grad():
                        r_other = lrp_instance.compute_relevance(self.current_selected_img, other_target_tensor)
                    other_maps.append(r_other)
                except Exception as e:
                    logger.error(f"    Error computing relevance for class {other_idx} for '{name}': {e}", exc_info=True)
            
            if other_maps:
                first_map_shape = other_maps[0].shape
                if all(m.shape == first_map_shape for m in other_maps):
                     stacked_others = torch.stack(other_maps)
                     r_avg_others = torch.mean(stacked_others, dim=0)
                     self.relevance_maps[name]['avg_others'] = r_avg_others
                     logger.info(f"      Avg Other Map Sum: {r_avg_others.sum():.4f}")
                else:
                     logger.error(f"    Cannot compute average map for '{name}': Inconsistent shapes.")
                     self.relevance_maps[name]['avg_others'] = None
            else:
                logger.warning(f"    No maps computed for other classes for '{name}'. Cannot average.")
                self.relevance_maps[name]['avg_others'] = None

            # 3. Compute Difference Map
            if r_target is not None and r_avg_others is not None:
                 try:
                     if r_target.shape == r_avg_others.shape:
                          r_diff = r_target - r_avg_others
                          self.relevance_maps[name]['difference'] = r_diff
                          logger.info(f"      Difference Map Sum: {r_diff.sum():.4f}")
                          # Add weighted difference map for final combination IF instance is active
                          if name in self.active_lrp_instances:
                              computed_diff_maps_for_combination.append(r_diff * weight)
                     else:
                          logger.error(f"    Cannot compute difference map for '{name}': Shape mismatch.")
                          self.relevance_maps[name]['difference'] = None
                 except Exception as e:
                     logger.error(f"    Error computing difference map for '{name}': {e}")
                     self.relevance_maps[name]['difference'] = None
            else:
                logger.warning(f"    Cannot compute difference map for '{name}'.")
                self.relevance_maps[name]['difference'] = None
        
        # Combine weighted difference maps (using only active instances)
        self.combined_relevance_map = self._combine_maps(computed_diff_maps_for_combination, "difference")
             
    def _combine_maps(self, maps_to_combine, map_type_name):
        """Helper to combine a list of maps using summation."""
        if not maps_to_combine:
            logger.warning(f"No maps provided to combine for type '{map_type_name}'.")
            return None
        # Check if only one map needs combining (can happen if only one instance active)
        if len(maps_to_combine) == 1:
             logger.info(f"Only one map to combine for '{map_type_name}', using it directly.")
             return maps_to_combine[0]
             
        logger.info(f"Combining {len(maps_to_combine)} {map_type_name} maps using weighted sum...")
        try:
             first_map_shape = maps_to_combine[0].shape
             if all(m.shape == first_map_shape for m in maps_to_combine):
                 stacked_maps = torch.stack(maps_to_combine)
                 combined_map = torch.sum(stacked_maps, dim=0)
                 logger.info(f"  Combined {map_type_name} map computed. Shape: {combined_map.shape}, Sum: {combined_map.sum():.4f}")
                 return combined_map
             else:
                 logger.error(f"Cannot combine {map_type_name} maps: Inconsistent shapes.")
                 return None
        except Exception as e:
             logger.error(f"Error during map combination for '{map_type_name}': {e}")
             return None

    def _plot_heatmap(self, ax, relevance_map, title, log_scale=False, vmin=None, vmax=None, apply_exp_scale=False, exp_scale_factor=2.0, positive_only=False, clamp_percentile=None):
        """Helper function to plot a single heatmap, potentially on a log scale or exponential scale."""
        if relevance_map is None:
            ax.set_title(f"{title} (Not Computed)")
            ax.imshow(np.zeros((self.args.image_size, self.args.image_size)), cmap='gray') # Placeholder
            ax.axis('off')
            return None

        heatmap_np = relevance_map.squeeze().cpu().numpy()
        
        # Handle multi-channel inputs (e.g., sum over color channels if needed)
        if len(heatmap_np.shape) == 3:
             if heatmap_np.shape[0] == 3: # Assume C, H, W
                  heatmap_np = heatmap_np.sum(axis=0)
             elif heatmap_np.shape[-1] == 3: # Assume H, W, C 
                  heatmap_np = heatmap_np.sum(axis=-1)
             elif heatmap_np.shape[0] == 1: # If squeezed but still has channel dim
                  heatmap_np = heatmap_np[0]
             else: # Unknown 3D shape, log warning and take first slice
                  logger.warning(f"Unexpected 3D heatmap shape {heatmap_np.shape} for '{title}'. Taking slice 0.")
                  heatmap_np = heatmap_np[0]
        
        plot_title = title
        heatmap_plot = None # Initialize to ensure it's assigned

        # --- Apply Clamping FIRST (if specified) ---
        if clamp_percentile is not None:
            try:
                lower_perc = np.percentile(heatmap_np, clamp_percentile)
                upper_perc = np.percentile(heatmap_np, 100 - clamp_percentile)
                heatmap_np = np.clip(heatmap_np, lower_perc, upper_perc)
                plot_title = f"{title}\n(Clamped {clamp_percentile}%) "
                logger.info(f"  Applying {clamp_percentile}% clamping to '{title}'. New range: [{heatmap_np.min():.4f}, {heatmap_np.max():.4f}]")
            except Exception as e:
                logger.warning(f"  Warning: Failed to apply percentile clamping for '{title}': {e}")
        # --- End Clamping ---

        # Apply scaling transformations AFTER clamping
        if apply_exp_scale:
            # Apply exponential scaling: sign(x) * abs(x)^factor
            # Use the potentially clamped heatmap_np for scaling
            heatmap_plot = np.sign(heatmap_np) * (np.abs(heatmap_np) ** exp_scale_factor)
            plot_title = f"{title}\n(Exp Scale: {exp_scale_factor:.1f})"
            # Determine vmin/vmax based on the scaled data
            abs_max_exp = np.max(np.abs(heatmap_plot))
            vmin = -abs_max_exp if abs_max_exp > 1e-9 else -1.0
            vmax = abs_max_exp if abs_max_exp > 1e-9 else 1.0
        elif log_scale:
            # Apply log scaling: sign(x) * log1p(abs(x))
            heatmap_plot = np.sign(heatmap_np) * np.log1p(np.abs(heatmap_np))
            plot_title = f"{title}\n(Log Scale)"
            if vmin is None or vmax is None:
                 abs_max_log = np.max(np.abs(heatmap_plot))
                 vmin = -abs_max_log if abs_max_log > 1e-9 else -1.0
                 vmax = abs_max_log if abs_max_log > 1e-9 else 1.0
        else:
            # No scaling, use original data
            heatmap_plot = heatmap_np
            if vmin is None or vmax is None:
                abs_max = np.max(np.abs(heatmap_plot))
                vmin = -abs_max if abs_max > 1e-9 else -1.0
                vmax = abs_max if abs_max > 1e-9 else 1.0
        
        # Apply positive-only filter *after* scaling but *before* determining color limits
        if positive_only:
            heatmap_plot = np.maximum(heatmap_plot, 0)
            plot_title += "\n(Positive Only)"
        
        # Determine vmin/vmax based on the (potentially filtered) scaled data
        if vmin is None or vmax is None: # Always recalculate if not provided, crucial for positive_only
            abs_max = np.max(np.abs(heatmap_plot))
            vmin = -abs_max if abs_max > 1e-9 else -1.0
            vmax = abs_max if abs_max > 1e-9 else 1.0
        
        im = ax.imshow(heatmap_plot, cmap='seismic', vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(plot_title, fontsize=9) # Slightly larger font now
        ax.axis('off')
        return im


    def visualize_results(self):
        """Visualize input, combined relevance, and exp-scaled combined relevance.
           Manages the figure for interactive display.
        """
        if self.combined_relevance_map is None:
             logger.info("No combined relevance map computed yet or computation failed. Cannot visualize.")
             # Don't clear the figure if it exists, just return
             # if self.fig:
             #      self.fig.clf()
             #      plt.draw()
             #      plt.pause(0.1)
             return

        # --- Figure Management --- 
        if self.fig is None or not plt.fignum_exists(self.fig.number):
            logger.info("Creating new figure for visualization (1x3 layout).")
            # Fixed layout: Input, Combined, Exp-Scaled Combined
            self.fig, self.axes = plt.subplots(1, 3, figsize=(18, 6))
        else:
            logger.info("Clearing existing figure for redraw.")
            self.fig.clf() # Clear the existing figure
            # Recreate axes on the cleared figure with the correct layout
            self.axes = self.fig.subplots(1, 3)
            
        if isinstance(self.axes, plt.Axes): # Handle case of single subplot (shouldn't happen with 1x3)
             self.axes = np.array([self.axes])
        self.axes = self.axes.flatten()

        # Determine titles based on mode
        if self.args.show_difference:
            combined_map_title = "Combined Difference"
        else:
            combined_map_title = "Combined Target"
        scaled_map_title = f"{combined_map_title}"

        # --- Calculate Log Scale Limits for the standard combined map --- 
        log_vmin, log_vmax = None, None
        map_for_limits = self.combined_relevance_map
        if self.args.log_scale and map_for_limits is not None:
            try:
                 map_np = map_for_limits.squeeze().cpu().numpy()
                 if len(map_np.shape) == 3:
                      if map_np.shape[0] in [1, 3]: map_np = map_np.sum(axis=0)
                      elif map_np.shape[-1] in [1, 3]: map_np = map_np.sum(axis=-1)
                      else: map_np = map_np[0] 
                 
                 log_map = np.sign(map_np) * np.log1p(np.abs(map_np))
                 abs_max_log = np.max(np.abs(log_map))
                 log_vmin = -abs_max_log if abs_max_log > 1e-9 else -1.0
                 log_vmax = abs_max_log if abs_max_log > 1e-9 else 1.0
                 logger.info(f"Using global log scale limits based on combined map: vmin={log_vmin:.4f}, vmax={log_vmax:.4f}")
            except Exception as e:
                 logger.error(f"Could not calculate log scale limits: {e}. Plots will use local limits.")
                 log_vmin, log_vmax = None, None # Ensure reset on error
        elif self.args.log_scale:
             logger.warning("Log scale requested, but combined map is missing.")

        # --- 1. Plot Input Image --- 
        ax_img = self.axes[0]
        img_display = self.current_selected_img.squeeze(0).cpu()

        # Log combined map stats before plotting
        if self.combined_relevance_map is not None:
            map_data = self.combined_relevance_map.cpu().numpy()
            logger.info(f"Combined Map Stats - Min: {np.min(map_data):.4f}, Max: {np.max(map_data):.4f}, Mean: {np.mean(map_data):.4f}, Median: {np.median(map_data):.4f}")

        if self.args.dataset == 'imagenet':
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_display = img_display * std + mean
            img_display = img_display.permute(1, 2, 0).numpy()
            img_display = np.clip(img_display, 0, 1) 
            ax_img.imshow(img_display)
        else: # MNIST
            ax_img.imshow(img_display.squeeze().numpy(), cmap='gray') 

        # Get True Label Name
        true_label_idx = self.labels[self.current_img_idx].item()
        true_class_name = "N/A"
        if self.class_names and 0 <= true_label_idx < len(self.class_names):
             full_name = self.class_names[true_label_idx]
             true_class_name = full_name.split(',')[0] 
        else:
             true_class_name = f"Index {true_label_idx}"

        # Get Target Label Name
        target_class_name = "N/A"
        if self.current_target_class_idx is not None:
            if self.class_names and 0 <= self.current_target_class_idx < len(self.class_names):
                 full_name = self.class_names[self.current_target_class_idx]
                 target_class_name = full_name.split(',')[0] 
            else:
                 target_class_name = f"Index {self.current_target_class_idx}"

        ax_img.set_title(f"Input Img {self.current_img_idx}\nTrue: {true_class_name} | Target: {target_class_name} ({self.current_target_class_prob:.3f})")
        ax_img.axis('off')

        # --- 2. Plot Combined Map --- 
        ax_comb = self.axes[1]
        im_comb = self._plot_heatmap(ax_comb,
                                      self.combined_relevance_map,
                                      title=combined_map_title,
                                      log_scale=self.args.log_scale,
                                      vmin=log_vmin, # Use calculated log limits if log scale
                                      vmax=log_vmax,
                                      clamp_percentile=self.args.clamp_percentile)

        # --- 3. Plot Exponentially Scaled Combined Map --- 
        ax_scaled = self.axes[2]
        # Note: positive_only is applied *within* _plot_heatmap after scaling
        im_scaled = self._plot_heatmap(ax_scaled,
                                       self.combined_relevance_map,
                                       title=scaled_map_title, 
                                       log_scale=False, # Do not apply log scale on top of exp scale
                                       apply_exp_scale=True,
                                       exp_scale_factor=self.args.exp_scale_factor,
                                       positive_only=self.args.positive_only,
                                       clamp_percentile=self.args.clamp_percentile) 

        # --- Add Colorbar --- 
        # Add colorbar for the *standard* combined map (im_comb)
        if im_comb:
             try:
                 cbar = self.fig.colorbar(im_comb, ax=ax_comb, fraction=0.046, pad=0.04)
                 label = 'Relevance'
                 if self.args.log_scale: label = 'Signed Log Relevance'
                 if self.args.show_difference: label += ' (Difference)'
                 cbar.set_label(label)
             except Exception as e:
                 logger.warning(f"Could not add colorbar for combined map: {e}")
        
        # Optionally, add a colorbar for the scaled map as well
        # if im_scaled:
        #     try:
        #         cbar_scaled = self.fig.colorbar(im_scaled, ax=ax_scaled, fraction=0.046, pad=0.04)
        #         cbar_scaled.set_label(f'Exp Scaled Relevance (Factor={self.args.exp_scale_factor:.1f})')
        #     except Exception as e:
        #         logger.warning(f"Could not add colorbar for scaled map: {e}")

        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        
        # --- Interactive Display --- 
        plt.draw() 
        plt.pause(0.1) # Allow GUI events to process

        # --- Save Figure (Optional) --- 
        if self.args.save_plots:
            output_dir = Path('lrp_combined_output')
            output_dir.mkdir(exist_ok=True)
            param_names = "_" + "_".join(sorted(self.active_lrp_instances.keys())) if self.active_lrp_instances else ""
            log_suffix = "_log" if self.args.log_scale else ""
            diff_suffix = "_diff" if self.args.show_difference else "_target"
            scale_suffix = f"_exp{self.args.exp_scale_factor:.1f}"
            filename = output_dir / f"img{self.current_img_idx}_true{true_label_idx}_target{self.current_target_class_idx}{param_names}{diff_suffix}{log_suffix}{scale_suffix}.png"
            try:
                self.fig.savefig(str(filename), dpi=150, bbox_inches='tight')
                logger.info(f"Saved visualization to {filename}")
            except Exception as e:
                logger.error(f"Error saving figure: {e}")


    def interactive_loop(self):
        """Runs the main interactive loop for selecting images and classes."""
        plt.ion() # Turn on interactive mode for matplotlib
        
        valid_indices_shown = set()
        
        while True:
            # Display top classes for current image
            valid_indices_shown = self._predict_and_show_top()
            
            # Get user input
            prompt = f"Enter class index to visualize (from list above), 'n' for next img (max {len(self.images)-1}), or 'q' to quit: "
            user_input = input(prompt).strip().lower()

            if user_input == 'q':
                logger.info("Quitting.")
                break
            elif user_input == 'n':
                self.current_img_idx += 1
                if self.current_img_idx >= len(self.images):
                    logger.info("Reached end of image batch, looping back to start.")
                    self.current_img_idx = 0
                continue # Loop to predict and show classes for the new image
            else:
                try:
                    selected_class_idx = int(user_input)
                    # Check if the selected index is valid for the model
                    if 0 <= selected_class_idx < self.num_classes:
                        # Optional: check if it was in the displayed list for better UX
                        # if selected_class_idx not in valid_indices_shown:
                        #     logger.warning(f"Index {selected_class_idx} was not in the top {self.args.top_n_display} predictions, but proceeding anyway.")
                            
                        # Set target class and compute/visualize
                        self.current_target_class_idx = selected_class_idx
                        logger.info(f"---> User selected index: {self.current_target_class_idx} <---") # Add logging here
                        self.compute_relevance_maps() # Computes based on current target/mode
                        self.visualize_results() # Updates the plot window
                    else:
                        print(f"Error: Class index {selected_class_idx} is out of range (0-{self.num_classes-1}).")
                except ValueError:
                    print("Error: Invalid input. Please enter a number, 'n', or 'q'.")
                except Exception as e:
                     logger.error(f"An error occurred during visualization or computation: {e}", exc_info=True)
                     print("An error occurred. Please check logs.")
                     
        plt.ioff() # Turn off interactive mode
        plt.close('all') # Close plot windows on exit
        logger.info("Interactive session ended.")



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Combined LRP Visualizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # --- Dataset and Model --- 
    ds_group = parser.add_argument_group('Dataset and Model')
    ds_group.add_argument('--dataset', type=str, default='imagenet', choices=AVAILABLE_DATASETS,
                        help=f'Dataset to use: {", ".join(AVAILABLE_DATASETS)}')
    ds_group.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory (required for ImageNet if not in default location)')
    ds_group.add_argument('--model_path', type=str, default=None,
                        help='Path to custom model (defaults to pretrained VGG16 for ImageNet, standard CNN for MNIST)')
    ds_group.add_argument('--batch_size', type=int, default=8, # Number of images to load for interaction
                        help='Number of images to load for interactive selection.')
    ds_group.add_argument('--subset_size', type=int, default=100, # Limit initial ImageNet scan if needed
                        help='Subset size for initial ImageNet scan (if needed by dataloader).')
    ds_group.add_argument('--image_size', type=int, default=224,
                        help='Image size for ImageNet preprocessing (resize)')
    
    # --- LRP Parameter Files --- 
    lrp_group = parser.add_argument_group('LRP Parameters')
    lrp_group.add_argument('--params-sensitivity', type=str, default=None,
                        help='Path to LRP parameters JSON file optimized for Average Sensitivity.')
    lrp_group.add_argument('--params-sparseness', type=str, default=None,
                        help='Path to LRP parameters JSON file optimized for Sparseness.')
    lrp_group.add_argument('--params-faithfulness', type=str, default=None,
                        help='Path to LRP parameters JSON file optimized for Faithfulness.')

    # --- Combination Weights --- 
    weight_group = parser.add_argument_group('Combination Weights')
    weight_group.add_argument('--weight-sensitivity', type=float, default=1.0,
                            help='Weight for the sensitivity map in the combination.')
    weight_group.add_argument('--weight-sparseness', type=float, default=1.0,
                            help='Weight for the sparseness map in the combination.')
    weight_group.add_argument('--weight-faithfulness', type=float, default=1.0,
                            help='Weight for the faithfulness map in the combination.')

    # --- Target Selection (Now for starting image) --- 
    target_group = parser.add_argument_group('Initial Target')
    target_group.add_argument('--img-idx', type=int, default=0,
                        help='Starting image index within the loaded batch.')
    # target_group.add_argument('--class-idx', type=int, default=None, # REMOVED
    #                     help='Target class index for LRP computation. If None, uses the top predicted class.')

    # --- Visualization Options --- 
    vis_group = parser.add_argument_group('Visualization')
    vis_group.add_argument('--top-n-display', type=int, default=10,
                         help='Number of top predictions to display for selection.')
    vis_group.add_argument('--log-scale', action='store_true',
                         help='Display relevance maps using a signed log scale (sign(x)*log1p(abs(x))). Recommended.')
    vis_group.add_argument('--show-difference', action='store_true',
                         help='Show the difference map (Target - Avg Others) instead of only the target map.')
    vis_group.add_argument('--diff-top-n', type=int, default=5,
                         help='Number of top other classes to average for the difference calculation.')
    vis_group.add_argument('--save-plots', action='store_true',
                         help='Save the generated plot for each selected class visualization.')
    vis_group.add_argument('--exp-scale-factor', type=float, default=2.0,
                         help='Exponent factor for the additional scaled relevance map plot (sign(x)*abs(x)**factor).')
    vis_group.add_argument('--positive-only', action='store_true',
                         help='Set all negative relevance values to zero before plotting.')
    vis_group.add_argument('--clamp-percentile', type=int, default=None,
                           help='Specify the percentile (e.g., 10 for 10th/90th) to clamp relevance heatmaps before visualization. No clamping if not set.')

    # --- Hardware --- 
    hw_group = parser.add_argument_group('Hardware')
    hw_group.add_argument('--device', type=str, default=None, choices=AVAILABLE_DEVICES,
                        help=f'Device to use: {", ".join(AVAILABLE_DEVICES)}. Auto-detects if None.')
    hw_group.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID to use when device is cuda.')
    
    args = parser.parse_args()
    
    # --- Post-processing and Validation --- 
    if args.data_dir is None and args.dataset == 'imagenet':
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent 
        args.data_dir = str(project_root / 'data' / 'ImageNet')
        logger.info(f"Defaulting ImageNet data directory to: {args.data_dir}")
        if not Path(args.data_dir).exists():
             logger.warning(f"Default ImageNet directory {args.data_dir} not found. Please specify --data_dir.")

    if not args.params_sensitivity and not args.params_sparseness and not args.params_faithfulness:
        parser.error("At least one --params-<metric> argument must be provided.")
        
    if args.show_difference and args.diff_top_n <= 0:
        parser.error("--diff-top-n must be positive when --show-difference is used.")
        
    if args.top_n_display <= 0:
        parser.error("--top-n-display must be positive.")

    if args.clamp_percentile is not None and not (0 < args.clamp_percentile < 50):
        parser.error("--clamp-percentile must be between 1 and 49.")

    return args

def main():
    """Main execution function."""
    try:
        args = parse_arguments()
        logger.info("Starting Interactive Combined LRP Visualizer...")
        args_dict = vars(args)
        logger.info(f"Arguments: {json.dumps(args_dict, indent=2)}") 
        
        visualizer = CombinedLRPVisualizer(args)
        visualizer.interactive_loop() # Call the new interactive loop
        
        logger.info("Visualization session ended.")
        return 0
        
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.") # Print to console
        logger.info("Execution interrupted by user.")
        # Ensure plots close cleanly if interrupted
        plt.close('all') 
        return 1
    except FileNotFoundError as e:
         logger.error(f"File not found error: {e}")
         return 1
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        plt.close('all') # Ensure plots close on other errors too
        return 1

if __name__ == "__main__":
    sys.exit(main())
