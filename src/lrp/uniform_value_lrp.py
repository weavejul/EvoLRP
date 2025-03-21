# lrp/uniform_value_lrp.py
import torch
import torch.nn as nn
from typing import List, Dict, Any
from .base import LRP
from .layers import AlphaBetaRule, EpsilonRule

class UniformValueLRP(LRP):
    def __init__(self, model: nn.Module, rule_types: List[str], rule_values: Dict[str, float]):
        """
        Layer-wise LRP where each layer can have a different rule type, but all layers with the same
        rule type share the same parameter value.
        
        Args:
            model: The neural network model
            rule_types: List of rule types for each layer (e.g., ['alpha_beta', 'epsilon', ...])
                        Must have the same length as the number of layers in the model
            rule_values: Dictionary mapping rule types to their parameter values
                        (e.g., {'alpha_beta': 2.0, 'epsilon': 0.1})
        """
        super().__init__(model)
        
        # Get total number of layers from features and classifier
        self.num_layers = len(list(model.features)) + len(list(model.classifier))
        
        # Validate rule types
        if len(rule_types) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} rule types, got {len(rule_types)}")
            
        # Validate rule types and make sure all required values are provided
        for rule_type in rule_types:
            if rule_type not in ['epsilon', 'alpha_beta', 'gamma']:
                raise ValueError(f"Unknown rule type: {rule_type}")
            if rule_type not in rule_values:
                raise ValueError(f"No parameter value provided for rule type: {rule_type}")
        
        self.rule_types = rule_types
        self.rule_values = rule_values
        
        # Store forward pass information
        self.forward_info = {}

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

    def compute_relevance(self, input_tensor):
        """
        Computes LRP relevance scores using the specified rule types with uniform values
        for each rule type.
        """
        # Ensure input is at least 4D (batch, channels, height, width)
        if len(input_tensor.shape) == 3:
            input_tensor = input_tensor.unsqueeze(0)
        
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
                # Store max pooling indices
                _, indices = torch.nn.functional.max_pool2d(
                    activations[-2], kernel_size=layer.kernel_size,
                    stride=layer.stride, padding=layer.padding,
                    return_indices=True
                )
                self.forward_info[f'layer_{i}']['indices'] = indices
            elif isinstance(layer, nn.BatchNorm2d):
                # Store batch norm statistics
                self.forward_info[f'layer_{i}']['mean'] = layer.running_mean
                self.forward_info[f'layer_{i}']['var'] = layer.running_var
        
        # Initialize relevance with the output
        relevance = current_activation.clone().detach()
        
        # Backward pass to compute relevance
        for i, (layer, activation) in enumerate(zip(reversed(layers), reversed(activations[:-1]))):
            # Get layer information
            layer_idx = self.num_layers - 1 - i
            layer_info = self.forward_info[f'layer_{layer_idx}']
            
            # Get the rule type for this layer
            rule_type = self.rule_types[layer_idx]
            
            # Create appropriate rule based on type with its uniform value
            if rule_type == 'alpha_beta':
                alpha = self.rule_values['alpha_beta']
                beta = alpha - 1  # Constraint: alpha = beta + 1
                rule = AlphaBetaRule(alpha=alpha, beta=beta)
                
                # Propagate relevance through layer
                relevance = rule.propagate(layer, relevance, activation)
                
            elif rule_type == 'epsilon':
                # For epsilon rule, param is the fraction of std to use as epsilon
                epsilon_fraction = self.rule_values['epsilon']
                rule = EpsilonRule(epsilon_fraction=epsilon_fraction)
                
                # Share forward information with the rule
                rule.forward_info = self.forward_info
                
                # Propagate relevance through layer
                relevance = rule.propagate(layer, relevance, activation)
                
            elif rule_type == 'gamma':
                gamma = self.rule_values['gamma']
                rule = GammaRule(gamma=gamma)
                                # Share forward information with the rule
                rule.forward_info = self.forward_info
                
                # Propagate relevance through layer
                relevance = rule.propagate(layer, relevance, activation)
            else:
                raise ValueError(f"Unknown rule type: {rule_type}")
            
            # Check if we need to reshape after propagating through a Linear layer
            if 'flattened' in layer_info and layer_info['flattened']:
                # Reshape back to the original shape before flattening
                pre_flatten_shape = layer_info['pre_flatten_shape']
                relevance = relevance.view(pre_flatten_shape)
            
            # Ensure relevance shape matches activation shape
            if relevance.shape != activation.shape:
                raise ValueError(f"Layer {i}: Relevance shape {relevance.shape} doesn't match activation shape {activation.shape}")
            
            # Store intermediate relevance for debugging/visualization
            self.forward_info[f'layer_{self.num_layers - 1 - i}']['relevance'] = relevance.clone().detach()
        
        # Normalize the final relevance map
        relevance = self._normalize_relevance(relevance)
        
        return relevance 