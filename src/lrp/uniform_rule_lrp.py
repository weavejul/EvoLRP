# lrp/uniform_rule_lrp.py
import torch
import torch.nn as nn
from .base import LRP
from .layers import AlphaBetaRule, EpsilonRule

class UniformRuleLRP(LRP):
    def __init__(self, model: nn.Module, rule_type: str, rule_params: torch.Tensor):
        """
        Uniform LRP where all layers use the same rule type but with different hyperparameters.
        
        Args:
            model: The neural network model
            rule_type: One of 'epsilon', 'alpha_beta', or 'gamma'
            rule_params: Tensor of shape (num_layers,) containing hyperparameters for each layer
        """
        super().__init__(model)
        self.rule_type = rule_type
        
        # Get total number of layers from features and classifier
        self.num_layers = len(list(model.features)) + len(list(model.classifier))
        
        # Store forward pass information
        self.forward_info = {}
        
        # Validate rule type
        if rule_type not in ['epsilon', 'alpha_beta', 'gamma']:
            raise ValueError(f"Unknown rule type: {rule_type}")
            
        # Validate rule parameters
        if len(rule_params) != self.num_layers:
            raise ValueError(f"Expected {self.num_layers} rule parameters, got {len(rule_params)}")
        
        # Validate and enforce rule-specific constraints
        if rule_type == 'alpha_beta':
            self.rule_params = self._validate_alpha_beta_params(rule_params)
        else:
            self.rule_params = rule_params

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
        Computes LRP relevance scores using the specified rule type with layer-specific hyperparameters.
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
                # print(f"Reshaping tensor from {current_activation.shape} for Linear layer")
                
                # Store original shape before flattening
                self.forward_info[f'layer_{i}']['pre_flatten_shape'] = current_activation.shape
                
                # Flatten preserving batch dimension (batch_size, channels*height*width)
                current_activation = current_activation.reshape(current_activation.shape[0], -1)
                
                self.forward_info[f'layer_{i}']['flattened'] = True
                self.forward_info[f'layer_{i}']['post_flatten_shape'] = current_activation.shape
                
                # print(f"Reshaped to {current_activation.shape}, linear expects {layer.weight.shape[1]} features")
            
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
            
            # Create appropriate rule based on type
            if self.rule_type == 'alpha_beta':
                alpha = self.rule_params[self.num_layers - 1 - i]
                beta = alpha - 1  # Constraint: alpha = beta + 1
                rule = AlphaBetaRule(alpha=alpha, beta=beta)
                
                # Propagate relevance through layer
                relevance = rule.propagate(layer, relevance, activation)
                
                # Check if we need to reshape after propagating through a Linear layer
                if 'flattened' in layer_info and layer_info['flattened']:
                    # Reshape back to the original shape before flattening
                    pre_flatten_shape = layer_info['pre_flatten_shape']
                    relevance = relevance.view(pre_flatten_shape)
                
                # Ensure relevance shape matches activation shape
                if relevance.shape != activation.shape:
                    raise ValueError(f"Layer {i}: Relevance shape {relevance.shape} doesn't match activation shape {activation.shape}")
            elif self.rule_type == 'epsilon':
                # For epsilon rule, param is the fraction of std to use as epsilon
                epsilon_fraction = self.rule_params[self.num_layers - 1 - i]
                rule = EpsilonRule(epsilon_fraction=epsilon_fraction)
                
                # Share forward information from UniformRuleLRP with the rule
                rule.forward_info = self.forward_info
                
                # Propagate relevance through layer
                relevance = rule.propagate(layer, relevance, activation)
                
                # Ensure relevance shape matches activation shape
                if relevance.shape != activation.shape:
                    raise ValueError(f"Layer {i}: Relevance shape {relevance.shape} doesn't match activation shape {activation.shape}")
            elif self.rule_type == 'gamma':
                gamma = self.rule_params[self.num_layers - 1 - i]
                # TODO: Implement gamma rule
                raise NotImplementedError("Gamma rule not implemented yet")
            else:
                raise ValueError(f"Unknown rule type: {self.rule_type}")
            
            # Store intermediate relevance for debugging/visualization
            self.forward_info[f'layer_{self.num_layers - 1 - i}']['relevance'] = relevance.clone().detach()
        
        # Normalize the final relevance map
        relevance = self._normalize_relevance(relevance)
        
        return relevance

    def _validate_alpha_beta_params(self, params):
        """
        Validate and ensure that alpha parameters are suitable for alpha-beta rule.
        
        Args:
            params: Raw parameters from the optimizer
        
        Returns:
            Validated parameters that respect alpha-beta relationship
        """
        # Ensure alpha is positive and beta = alpha - 1
        if torch.min(params) < 1.0:
            print(f"Warning: Some alpha parameters are < 1.0. Constraining to ≥ 1.0")
            params = torch.clamp(params, min=1.0)
            
        return params