import torch
import torch.nn as nn

class AlphaBetaRule:
    def __init__(self, alpha: float, beta: float, eps: float = 1e-9):
        """
        Initialize the alpha-beta rule.
        
        Args:
            alpha: Parameter for positive contributions
            beta: Parameter for negative contributions (should be alpha - 1)
            eps: Small constant for numerical stability
        """
        # Validate alpha-beta relationship
        if not abs(alpha - beta - 1) < 1e-6:
            raise ValueError(f"Beta ({beta}) should be alpha - 1 ({alpha - 1})")
            
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # Store forward pass information
        self.forward_info = {}

    def _ensure_shape_match(self, tensor: torch.Tensor, expected_shape: tuple, name: str) -> None:
        """Ensure tensor shape matches expected shape."""
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tensor.shape} doesn't match expected shape {expected_shape}")

    def _safe_divide(self, numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
        """Safely divide two tensors with numerical stability."""
        return numerator / (denominator + self.eps)

    def _handle_bias(self, relevance: torch.Tensor, input_tensor: torch.Tensor, 
                    bias: torch.Tensor, bias_shape: tuple) -> torch.Tensor:
        """Handle bias term with proper reshaping and numerical stability."""
        bias = bias.view(bias_shape)
        bias_relevance = relevance * self._safe_divide(input_tensor, input_tensor + bias)
        return relevance + bias_relevance

    def propagate(self, layer: nn.Module, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Propagate relevance through a layer using the alpha-beta rule.
        
        Args:
            layer: The layer to propagate through
            relevance: The relevance from the next layer
            input_tensor: The input to the current layer
            
        Returns:
            The propagated relevance
        """
        # Dictionary mapping layer types to their propagation methods
        layer_handlers = {
            nn.Linear: self._propagate_linear,
            nn.Conv2d: self._propagate_conv2d,
            nn.Conv1d: self._propagate_conv1d,
            nn.Conv3d: self._propagate_conv3d,
            nn.ReLU: self._propagate_relu,
            nn.LeakyReLU: self._propagate_leaky_relu,
            nn.BatchNorm2d: self._propagate_batch_norm,
            nn.BatchNorm1d: self._propagate_batch_norm,
            nn.BatchNorm3d: self._propagate_batch_norm,
            nn.MaxPool2d: self._propagate_max_pool,
            nn.AvgPool2d: self._propagate_avg_pool,
            nn.Flatten: self._propagate_flatten,
            nn.Unflatten: self._propagate_unflatten,
            nn.Dropout: self._propagate_dropout,
            nn.Identity: self._propagate_identity,
        }

        # Get the appropriate handler for the layer type
        layer_type = type(layer)
        handler = layer_handlers.get(layer_type)
        
        if handler:
            return handler(layer, relevance, input_tensor)
        else:
            # For unknown layer types, try to handle them based on their structure
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                # If it has weights and bias, treat it like a linear layer
                return self._propagate_linear(layer, relevance, input_tensor)
            elif isinstance(layer, (nn.Sequential, nn.ModuleList)):
                # For sequential layers, propagate through each sublayer
                current_relevance = relevance
                current_input = input_tensor
                for sublayer in reversed(list(layer)):  # Propagate in reverse order
                    current_relevance = self.propagate(sublayer, current_relevance, current_input)
                    current_input = sublayer(current_input)  # Update input for next layer
                return current_relevance
            else:
                # For other layers, use identity propagation with a warning
                print(f"Warning: Using identity propagation for unknown layer type {layer_type}")
                return self._propagate_identity(layer, relevance, input_tensor)

    def _propagate_linear(self, layer: nn.Linear, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a linear layer."""
        # Get positive and negative weights
        pos_weights = torch.clamp(layer.weight, min=0.0)
        neg_weights = torch.clamp(layer.weight, max=0.0)
        
        # Handle both batched and unbatched inputs
        if len(relevance.shape) == 1:
            relevance = relevance.unsqueeze(0)
            input_tensor = input_tensor.unsqueeze(0)
        
        # Compute contributions using weight transpose
        pos_contrib = torch.matmul(relevance, pos_weights)
        neg_contrib = torch.matmul(relevance, neg_weights)
        
        # Combine contributions using alpha-beta rule
        relevance = self.alpha * pos_contrib + self.beta * neg_contrib
        
        # Handle bias if present
        if layer.bias is not None:
            # We skip bias handling for linear layers as it doesn't contribute to relevance
            # in a meaningful way during backward propagation
            pass
        
        # Remove batch dimension if input was unbatched
        if len(input_tensor.shape) == 1:
            relevance = relevance.squeeze(0)
        
        return relevance

    def _propagate_conv2d(self, layer: nn.Conv2d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a 2D convolutional layer."""
        # Check channel dimension only
        if relevance.shape[1] != layer.out_channels:
            raise ValueError(f"Relevance channels {relevance.shape[1]} doesn't match layer output channels {layer.out_channels}")
            
        # Get positive and negative weights
        pos_weights = torch.clamp(layer.weight, min=0.0)
        neg_weights = torch.clamp(layer.weight, max=0.0)
        
        # Compute positive and negative contributions using transposed convolution
        pos_contrib = torch.nn.functional.conv_transpose2d(
            relevance, pos_weights, stride=layer.stride, padding=layer.padding,
            groups=layer.groups
        )
        neg_contrib = torch.nn.functional.conv_transpose2d(
            relevance, neg_weights, stride=layer.stride, padding=layer.padding,
            groups=layer.groups
        )
        
        # Combine contributions using alpha-beta rule
        relevance = self.alpha * pos_contrib + self.beta * neg_contrib
        
        # Handle bias if present
        if layer.bias is not None:
            # Skip bias handling for conv layers as it's not meaningful for relevance propagation
            pass
            
        return relevance

    def _propagate_conv1d(self, layer: nn.Conv1d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a 1D convolutional layer."""
        # Ensure input shapes match
        if relevance.shape[1] != layer.out_channels:
            raise ValueError(f"Relevance channels {relevance.shape[1]} doesn't match layer output channels {layer.out_channels}")
            
        # Get positive and negative weights
        pos_weights = torch.clamp(layer.weight, min=0.0)
        neg_weights = torch.clamp(layer.weight, max=0.0)
        
        # Compute positive and negative contributions using transposed convolution
        pos_contrib = torch.nn.functional.conv_transpose1d(
            relevance, pos_weights, stride=layer.stride, padding=layer.padding,
            groups=layer.groups
        )
        neg_contrib = torch.nn.functional.conv_transpose1d(
            relevance, neg_weights, stride=layer.stride, padding=layer.padding,
            groups=layer.groups
        )
        
        # Combine contributions using alpha-beta rule
        relevance = self.alpha * pos_contrib + self.beta * neg_contrib
        
        # Handle bias if present
        if layer.bias is not None:
            # For bias, we distribute the relevance proportionally to the input
            bias_relevance = relevance * (input_tensor / (input_tensor + layer.bias.view(1, -1, 1) + 1e-9))
            relevance = relevance + bias_relevance
            
        return relevance

    def _propagate_conv3d(self, layer: nn.Conv3d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a 3D convolutional layer."""
        # Ensure input shapes match
        if relevance.shape[1] != layer.out_channels:
            raise ValueError(f"Relevance channels {relevance.shape[1]} doesn't match layer output channels {layer.out_channels}")
            
        # Get positive and negative weights
        pos_weights = torch.clamp(layer.weight, min=0.0)
        neg_weights = torch.clamp(layer.weight, max=0.0)
        
        # Compute positive and negative contributions using transposed convolution
        pos_contrib = torch.nn.functional.conv_transpose3d(
            relevance, pos_weights, stride=layer.stride, padding=layer.padding,
            groups=layer.groups
        )
        neg_contrib = torch.nn.functional.conv_transpose3d(
            relevance, neg_weights, stride=layer.stride, padding=layer.padding,
            groups=layer.groups
        )
        
        # Combine contributions using alpha-beta rule
        relevance = self.alpha * pos_contrib + self.beta * neg_contrib
        
        # Handle bias if present
        if layer.bias is not None:
            # For bias, we distribute the relevance proportionally to the input
            bias_relevance = relevance * (input_tensor / (input_tensor + layer.bias.view(1, -1, 1, 1, 1) + 1e-9))
            relevance = relevance + bias_relevance
            
        return relevance

    def _propagate_relu(self, layer: nn.ReLU, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a ReLU layer."""
        mask = (input_tensor > 0).float()
        return relevance * mask

    def _propagate_leaky_relu(self, layer: nn.LeakyReLU, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a LeakyReLU layer."""
        mask = torch.where(input_tensor > 0, torch.ones_like(input_tensor), 
                         layer.negative_slope * torch.ones_like(input_tensor))
        return relevance * mask

    def _propagate_batch_norm(self, layer: nn.BatchNorm2d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a batch normalization layer."""
        # Get running statistics with proper reshaping
        mean = layer.running_mean.view(1, -1, 1, 1)
        var = layer.running_var.view(1, -1, 1, 1)
        gamma = layer.weight.view(1, -1, 1, 1) if layer.affine else 1.0
        beta = layer.bias.view(1, -1, 1, 1) if layer.affine else 0.0
        
        # Compute normalized input with numerical stability
        normalized_input = (input_tensor - mean) / (torch.sqrt(var + layer.eps))
        
        # Apply alpha-beta rule to the learnable parameters
        if layer.affine:
            pos_gamma = torch.clamp(gamma, min=0.0)
            neg_gamma = torch.clamp(gamma, max=0.0)
            relevance = self.alpha * (relevance * pos_gamma) + self.beta * (relevance * neg_gamma)
        
        # Handle bias if present
        if layer.affine and layer.bias is not None:
            relevance = self._handle_bias(relevance, normalized_input, beta, (1, -1, 1, 1))
            
        return relevance

    def _propagate_max_pool(self, layer: nn.MaxPool2d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a max pooling layer."""
        # Get the stored indices from forward pass
        indices = self.forward_info.get('max_pool_indices')
        if indices is None:
            # If no indices stored, we need to compute them
            _, indices = torch.nn.functional.max_pool2d(
                input_tensor, kernel_size=layer.kernel_size, stride=layer.stride,
                padding=layer.padding, return_indices=True
            )
            self.forward_info['max_pool_indices'] = indices
        
        # Ensure output size matches input size
        output_size = input_tensor.shape[2:]
        
        # Use max_unpool2d for backward propagation
        return torch.nn.functional.max_unpool2d(
            relevance, indices, kernel_size=layer.kernel_size, stride=layer.stride,
            padding=layer.padding, output_size=output_size
        )

    def _propagate_avg_pool(self, layer: nn.AvgPool2d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through an average pooling layer."""
        # Handle tuple or single value parameters
        kernel_size = layer.kernel_size if isinstance(layer.kernel_size, tuple) else (layer.kernel_size, layer.kernel_size)
        stride = layer.stride if isinstance(layer.stride, tuple) else (layer.stride, layer.stride)
        padding = layer.padding if isinstance(layer.padding, tuple) else (layer.padding, layer.padding)
        
        # Create a mask to distribute relevance evenly
        mask = torch.ones_like(input_tensor)
        mask = torch.nn.functional.avg_pool2d(mask, kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Ensure proper scaling with numerical stability
        mask = torch.nn.functional.interpolate(mask, size=input_tensor.shape[2:], mode='nearest')
        mask = self._safe_divide(mask, mask.sum(dim=(2, 3), keepdim=True))
        
        # Distribute relevance evenly
        return relevance * mask

    def _propagate_flatten(self, layer: nn.Flatten, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a flatten layer."""
        return relevance.view(input_tensor.shape)

    def _propagate_unflatten(self, layer: nn.Unflatten, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through an unflatten layer."""
        return relevance.view(input_tensor.shape)

    def _propagate_dropout(self, layer: nn.Dropout, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a dropout layer."""
        # During inference, dropout is identity
        return relevance

    def _propagate_identity(self, layer: nn.Identity, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through an identity layer."""
        return relevance
