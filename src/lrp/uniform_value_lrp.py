# lrp/uniform_value_lrp.py
import torch
import torch.nn as nn
from typing import List, Dict, Any
from .base import LRP
from .layers import AlphaBetaRule, EpsilonRule, GammaRule, ZeroRule

class UniformValueLRP(LRP):
    def __init__(self, model: nn.Module, rule_types: List[str], rule_values: Dict[str, float], correct_class_only: bool = False):
        """
        Layer-wise LRP where each layer can have a different rule type, but all layers with the same
        rule type share the same parameter value.
        
        Args:
            model: The neural network model
            rule_types: List of rule types for each layer (e.g., ['alpha_beta', 'epsilon', ...])
                        Must have the same length as the number of layers in the model
            rule_values: Dictionary mapping rule types to their parameter values
                        (e.g., {'alpha_beta': 2.0, 'epsilon': 0.1})
            correct_class_only: If True, calculate relevance only for the correct class
        """
        super().__init__(model)
        
        # Get total number of layers from features and classifier
        self.num_layers = len(list(model.features)) + len(list(model.classifier))
        
        # Validate rule types
        if len(rule_types) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} rule types, got {len(rule_types)}")
            
        # Validate rule types and make sure all required values are provided
        for rule_type in rule_types:
            if rule_type not in ['epsilon', 'alpha_beta', 'gamma', 'zero']:
                raise ValueError(f"Unknown rule type: {rule_type}")
            if rule_type not in rule_values:
                raise ValueError(f"No parameter value provided for rule type: {rule_type}")
        
        self.rule_types = rule_types
        self.rule_values = rule_values
        self.correct_class_only = correct_class_only
        
        # Store forward pass information
        self.forward_info = {}
        
        # Track MaxPool layers specifically
        self.max_pool_indices = {}

    def _normalize_relevance(self, relevance: torch.Tensor) -> torch.Tensor:
        """
        Normalize the relevance map to [-1, 1] range.
        
        Args:
            relevance: The relevance map to normalize
            
        Returns:
            Normalized relevance map
        """
        # Handle batched inputs
        if len(relevance.shape) == 4:  # (B, C, H, W)
            # Normalize each sample independently
            for i in range(relevance.shape[0]):
                max_val = torch.max(torch.abs(relevance[i]))
                if max_val > 0:
                    relevance[i] = relevance[i] / max_val
        else:  # (C, H, W)
            max_val = torch.max(torch.abs(relevance))
            if max_val > 0:
                relevance = relevance / max_val
        return relevance

    def compute_relevance(self, input_tensor, labels=None):
        """
        Computes LRP relevance scores using the specified rule types with uniform values
        for each rule type.
        
        Args:
            input_tensor: The input tensor to compute relevance for
            labels: Optional class labels, required if correct_class_only=True
        """
        # Check if labels are provided when needed
        if self.correct_class_only and labels is None:
            raise ValueError("Labels must be provided when correct_class_only=True")
            
        # Ensure input is at least 4D (batch, channels, height, width)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
            if labels is not None and not torch.is_tensor(labels):
                labels = torch.tensor([labels], device=input_tensor.device)
            elif labels is not None:
                labels = labels.unsqueeze(0)
        
        # Store input shape for later use
        self.forward_info['input_shape'] = input_tensor.shape
        
        # Forward pass to collect activations and other information
        activations = []
        current_activation = input_tensor.clone().detach()
        
        # Store initial activation
        activations.append(current_activation)
        
        # Get all layers in order
        layers = []
        if hasattr(self.model, 'features'):
            layers.extend(list(self.model.features))
            
            # For VGG and similar models, there's typically a flattening operation
            # between features and classifier that isn't a module
            self.forward_info['features_output_shape'] = current_activation.shape
        
        if hasattr(self.model, 'classifier'):
            layers.extend(list(self.model.classifier))
        
        # Track MaxPool layers by index for a more robust approach
        max_pool_count = 0
        
        # Forward pass through each layer
        for i, layer in enumerate(layers):
            # Store layer information
            self.forward_info[f'layer_{i}'] = {
                'type': type(layer),
                'input_shape': current_activation.shape
            }
            
            # Special case for transition from features to classifier in VGG-like models
            if (i > 0 and 
                isinstance(layer, nn.Linear) and
                len(current_activation.shape) > 2):  # If tensor has more than 2 dimensions
                
                # This is likely the first linear layer after convolutional features
                
                # Store original shape before flattening
                self.forward_info[f'layer_{i}']['pre_flatten_shape'] = current_activation.shape
                
                # Flatten preserving batch dimension (batch_size, channels*height*width)
                current_activation = current_activation.reshape(current_activation.shape[0], -1)
                
                self.forward_info[f'layer_{i}']['flattened'] = True
                self.forward_info[f'layer_{i}']['post_flatten_shape'] = current_activation.shape
            
            # Forward pass through layer
            current_activation = layer(current_activation)
            activations.append(current_activation)
            
            # Store additional layer-specific information
            if isinstance(layer, nn.MaxPool2d):
                # Store max pooling indices and input shape for unpooling
                input_for_pooling = activations[-2]
                _, indices = torch.nn.functional.max_pool2d(
                    input_for_pooling, kernel_size=layer.kernel_size,
                    stride=layer.stride, padding=layer.padding,
                    return_indices=True
                )
                
                # Store layer-specific information with MaxPool index for robust tracking
                max_pool_id = max_pool_count
                max_pool_count += 1
                
                # Store both in layer info and in MaxPool-specific tracker
                self.forward_info[f'layer_{i}']['indices'] = indices
                self.forward_info[f'layer_{i}']['input_shape_for_unpooling'] = input_for_pooling.shape
                self.forward_info[f'layer_{i}']['max_pool_id'] = max_pool_id
                
                # Also track in dedicated dictionary for easier access
                self.max_pool_indices[max_pool_id] = {
                    'indices': indices,
                    'input_shape': input_for_pooling.shape,
                    'layer_idx': i
                }
                
            elif isinstance(layer, nn.BatchNorm2d):
                # Store batch norm statistics
                self.forward_info[f'layer_{i}']['mean'] = layer.running_mean
                self.forward_info[f'layer_{i}']['var'] = layer.running_var
        
        # Initialize relevance with the output
        relevance = current_activation.clone().detach()
        
        # Check for NaNs in initial relevance
        if torch.isnan(relevance).any():
            print("Warning: NaN values detected in initial relevance")
            relevance = torch.nan_to_num(relevance, nan=0.0)
        
        # If we want to focus on correct class only
        if self.correct_class_only and labels is not None:
            # Create a mask with zeros and set only the correct class to the activation value
            batch_size = relevance.shape[0]
            num_classes = relevance.shape[1]
            mask = torch.zeros_like(relevance)
            
            # Set only the correct class relevance
            for i in range(batch_size):
                mask[i, labels[i]] = relevance[i, labels[i]]
                
            relevance = mask
        
        # MaxPool layer counter for backpropagation
        max_pool_count_reverse = 0
        
        # Backward pass to compute relevance
        for i, (layer, activation) in enumerate(zip(reversed(layers), reversed(activations[:-1]))):
            # Get layer information
            layer_idx = self.num_layers - 1 - i
            layer_info = self.forward_info[f'layer_{layer_idx}']
            
            # Get the rule type for this layer
            rule_type = self.rule_types[layer_idx]
            
            try:
                # Prepare a clean rule-specific forward_info for each layer
                rule_forward_info = {}
                
                # If it's a MaxPool layer, ensure we provide correct layer-specific indices
                if isinstance(layer, nn.MaxPool2d):
                    # Get the specific MaxPool ID for this layer
                    if 'max_pool_id' in layer_info:
                        max_pool_id = layer_info['max_pool_id']
                        max_pool_data = self.max_pool_indices[max_pool_id]
                        
                        # Add layer-specific MaxPool data to rule_forward_info
                        rule_forward_info['max_pool_indices'] = max_pool_data['indices']
                        rule_forward_info['max_pool_input_shape'] = max_pool_data['input_shape']
                        rule_forward_info['output_size'] = max_pool_data['input_shape']
                
                # Create appropriate rule based on type with its uniform value
                if rule_type == 'alpha_beta':
                    alpha = self.rule_values['alpha_beta']
                    beta = alpha - 1  # Constraint: alpha = beta + 1
                    rule = AlphaBetaRule(alpha=alpha, beta=beta)
                    
                    # Share only the layer-specific forward info
                    rule.forward_info = rule_forward_info
                    
                    # Propagate relevance through layer
                    with torch.no_grad():
                        relevance = rule.propagate(layer, relevance, activation)
                    
                    # Check for NaNs after each layer's relevance propagation
                    if torch.isnan(relevance).any():
                        print(f"Warning: NaN values detected after layer {layer_idx} ({type(layer).__name__}) with rule {rule_type}")
                        relevance = torch.nan_to_num(relevance, nan=0.0)
                    
                elif rule_type == 'epsilon':
                    # For epsilon rule, param is the fraction of std to use as epsilon
                    epsilon_fraction = self.rule_values['epsilon']
                    rule = EpsilonRule(epsilon_fraction=epsilon_fraction)
                    
                    # Share forward information with the rule but only layer-specific
                    rule.forward_info = rule_forward_info
                    
                    # If linear, flatten the activation
                    if isinstance(layer, nn.Linear) and 'flattened' in layer_info and layer_info['flattened']:
                        activation = activation.view(activation.shape[0], -1)
                    
                    # Propagate relevance through layer
                    with torch.no_grad():
                        relevance = rule.propagate(layer, relevance, activation)
                    
                    # Check for NaNs after each layer's relevance propagation
                    if torch.isnan(relevance).any():
                        print(f"Warning: NaN values detected after layer {layer_idx} ({type(layer).__name__}) with rule {rule_type}")
                        relevance = torch.nan_to_num(relevance, nan=0.0)
                    
                elif rule_type == 'gamma':
                    gamma = self.rule_values['gamma']
                    rule = GammaRule(gamma=gamma)
                    
                    # Share only layer-specific forward information with the rule
                    rule.forward_info = rule_forward_info
                    
                    # If linear, flatten the activation
                    if isinstance(layer, nn.Linear) and 'flattened' in layer_info and layer_info['flattened']:
                        activation = activation.view(activation.shape[0], -1)
                    
                    # Propagate relevance through layer
                    with torch.no_grad():
                        relevance = rule.propagate(layer, relevance, activation)
                    
                    # Check for NaNs after each layer's relevance propagation
                    if torch.isnan(relevance).any():
                        print(f"Warning: NaN values detected after layer {layer_idx} ({type(layer).__name__}) with rule {rule_type}")
                        relevance = torch.nan_to_num(relevance, nan=0.0)
                elif rule_type == 'zero':
                    rule = ZeroRule()
                    # Share forward information with the rule but only layer-specific
                    rule.forward_info = rule_forward_info
                    
                    # If linear, flatten the activation
                    if isinstance(layer, nn.Linear) and 'flattened' in layer_info and layer_info['flattened']:
                        activation = activation.view(activation.shape[0], -1)
                    
                    # Propagate relevance through layer
                    with torch.no_grad():
                        relevance = rule.propagate(layer, relevance, activation)
                    
                    # Check for NaNs after each layer's relevance propagation
                    if torch.isnan(relevance).any():
                        print(f"Warning: NaN values detected after layer {layer_idx} ({type(layer).__name__}) with rule {rule_type}")
                        relevance = torch.nan_to_num(relevance, nan=0.0)
                else:
                    raise ValueError(f"Unknown rule type: {rule_type}")
                
                # Cleanup to free memory
                del rule
                
                # Check if we need to reshape after propagating through a Linear layer
                if 'flattened' in layer_info and layer_info['flattened']:
                    # Reshape back to the original shape before flattening
                    pre_flatten_shape = layer_info['pre_flatten_shape']
                    relevance = relevance.view(pre_flatten_shape)
                    activation = activation.view(pre_flatten_shape)
                
                # Ensure relevance shape matches activation shape
                if relevance.shape != activation.shape:
                    raise ValueError(f"Layer {i}: Relevance shape {relevance.shape} doesn't match activation shape {activation.shape}")
                
                # Store intermediate relevance for debugging/visualization
                self.forward_info[f'layer_{self.num_layers - 1 - i}']['relevance'] = relevance.clone().detach()
                
            except Exception as e:
                print(f"Error in layer {layer_idx} ({type(layer).__name__}) with rule {rule_type}: {str(e)}")
                raise
        
        # Final NaN check before normalization
        if torch.isnan(relevance).any():
            print("Warning: NaN values detected in final relevance before normalization")
            relevance = torch.nan_to_num(relevance, nan=0.0)
        
        # Normalize the final relevance map
        relevance = self._normalize_relevance(relevance)
        
        # Release memory
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return relevance 