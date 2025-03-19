import torch
import torch.nn as nn

class EpsilonRule:
    def __init__(self, epsilon_fraction=0.25, min_epsilon=1e-9):
        """
        Initialize the LRP-ε rule with dynamic epsilon based on activation statistics.
        
        Args:
            epsilon_fraction: Fraction of standard deviation to use for epsilon (e.g., 0.25 = 25% of std)
            min_epsilon: Minimum epsilon value for numerical stability
        """
        self.epsilon_fraction = epsilon_fraction
        self.min_epsilon = min_epsilon
        # Store forward pass information
        self.forward_info = {}

    def _compute_epsilon(self, input_tensor):
        """
        Compute epsilon value as a fraction of the standard deviation of input activations.
        
        Args:
            input_tensor: Input activations to the layer
            
        Returns:
            epsilon: The computed epsilon value
        """
        # Calculate standard deviation of activations
        std = torch.std(input_tensor)
        
        # Set epsilon to the specified fraction of standard deviation
        epsilon = self.epsilon_fraction * std
        
        # Ensure epsilon is at least the minimum value for numerical stability
        epsilon = torch.max(epsilon, torch.tensor(self.min_epsilon))
        
        return epsilon

    def _ensure_shape_match(self, tensor: torch.Tensor, expected_shape: tuple, name: str) -> None:
        """Ensure tensor shape matches expected shape."""
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape {tensor.shape} doesn't match expected shape {expected_shape}")

    def propagate(self, layer: nn.Module, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Propagate relevance through a layer using the epsilon rule.
        
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
        """Propagate relevance through a linear layer using the epsilon rule."""
        # Handle both batched and unbatched inputs
        if len(relevance.shape) == 1:
            relevance = relevance.unsqueeze(0)
            input_tensor = input_tensor.unsqueeze(0)
        
        # Compute dynamic epsilon based on input tensor statistics
        epsilon = self._compute_epsilon(input_tensor)
        
        # Compute forward activation with layer weights
        z = torch.matmul(input_tensor, layer.weight.t())
        if layer.bias is not None:
            z += layer.bias.unsqueeze(0)
        
        # Compute the stabilized denominator with epsilon
        s = relevance / (z + epsilon * torch.sign(z))
        
        # Calculate input-level relevance
        c = torch.matmul(s, layer.weight)
        relevance = input_tensor * c
        
        # Remove batch dimension if input was unbatched
        if len(input_tensor.shape) == 1:
            relevance = relevance.squeeze(0)
        
        return relevance

    def _propagate_conv2d(self, layer: nn.Conv2d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a 2D convolutional layer using the epsilon rule."""
        # Calculate dynamic epsilon from input statistics
        epsilon = self._compute_epsilon(input_tensor)
        
        # Compute forward activation (z)
        z = torch.nn.functional.conv2d(
            input_tensor, layer.weight, bias=layer.bias,
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups
        )
        
        # Add small epsilon to stabilize division (maintaining sign)
        stabilized_z = z + epsilon * torch.sign(z)
        
        # Create a function that computes input relevance when given the output relevance
        def grad_fn(grad_output):
            # Distribute relevance proportionally to input contributions
            return torch.nn.grad.conv2d_input(
                input_tensor.shape, layer.weight, grad_output,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups
            )
        
        # Compute relevance to propagate backward
        backward_relevance = relevance / stabilized_z
        
        # Compute input-level relevance
        input_relevance = grad_fn(backward_relevance)
        
        # Scale by input values to get final relevance
        input_relevance = input_tensor * input_relevance
        
        return input_relevance

    def _propagate_conv1d(self, layer: nn.Conv1d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a 1D convolutional layer using the epsilon rule."""
        # Calculate dynamic epsilon from input statistics
        epsilon = self._compute_epsilon(input_tensor)
        
        # Compute forward activation (z)
        z = torch.nn.functional.conv1d(
            input_tensor, layer.weight, bias=layer.bias,
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups
        )
        
        # Add small epsilon to stabilize division (maintaining sign)
        stabilized_z = z + epsilon * torch.sign(z)
        
        # Create a function that computes input relevance when given the output relevance
        def grad_fn(grad_output):
            # Distribute relevance proportionally to input contributions
            return torch.nn.grad.conv1d_input(
                input_tensor.shape, layer.weight, grad_output,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups
            )
        
        # Compute relevance to propagate backward
        backward_relevance = relevance / stabilized_z
        
        # Compute input-level relevance
        input_relevance = grad_fn(backward_relevance)
        
        # Scale by input values to get final relevance
        input_relevance = input_tensor * input_relevance
        
        return input_relevance

    def _propagate_conv3d(self, layer: nn.Conv3d, relevance: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        """Propagate relevance through a 3D convolutional layer using the epsilon rule."""
        # Calculate dynamic epsilon from input statistics
        epsilon = self._compute_epsilon(input_tensor)
        
        # Compute forward activation (z)
        z = torch.nn.functional.conv3d(
            input_tensor, layer.weight, bias=layer.bias,
            stride=layer.stride, padding=layer.padding,
            dilation=layer.dilation, groups=layer.groups
        )
        
        # Add small epsilon to stabilize division (maintaining sign)
        stabilized_z = z + epsilon * torch.sign(z)
        
        # Create a function that computes input relevance when given the output relevance
        def grad_fn(grad_output):
            # Distribute relevance proportionally to input contributions
            return torch.nn.grad.conv3d_input(
                input_tensor.shape, layer.weight, grad_output,
                stride=layer.stride, padding=layer.padding,
                dilation=layer.dilation, groups=layer.groups
            )
        
        # Compute relevance to propagate backward
        backward_relevance = relevance / stabilized_z
        
        # Compute input-level relevance
        input_relevance = grad_fn(backward_relevance)
        
        # Scale by input values to get final relevance
        input_relevance = input_tensor * input_relevance
        
        return input_relevance

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
        """Propagate relevance through a batch normalization layer using the epsilon rule."""
        # Calculate dynamic epsilon from input statistics
        epsilon = self._compute_epsilon(input_tensor)
        
        # Reshape running statistics for broadcasting
        if isinstance(layer, nn.BatchNorm1d):
            running_mean = layer.running_mean.view(1, -1, 1)
            running_var = layer.running_var.view(1, -1, 1)
            weight = layer.weight.view(1, -1, 1) if layer.affine else torch.ones_like(running_mean)
            bias = layer.bias.view(1, -1, 1) if layer.affine else torch.zeros_like(running_mean)
        elif isinstance(layer, nn.BatchNorm2d):
            running_mean = layer.running_mean.view(1, -1, 1, 1)
            running_var = layer.running_var.view(1, -1, 1, 1)
            weight = layer.weight.view(1, -1, 1, 1) if layer.affine else torch.ones_like(running_mean)
            bias = layer.bias.view(1, -1, 1, 1) if layer.affine else torch.zeros_like(running_mean)
        elif isinstance(layer, nn.BatchNorm3d):
            running_mean = layer.running_mean.view(1, -1, 1, 1, 1)
            running_var = layer.running_var.view(1, -1, 1, 1, 1)
            weight = layer.weight.view(1, -1, 1, 1, 1) if layer.affine else torch.ones_like(running_mean)
            bias = layer.bias.view(1, -1, 1, 1, 1) if layer.affine else torch.zeros_like(running_mean)
        
        # Compute normalized input
        x_norm = (input_tensor - running_mean) / torch.sqrt(running_var + layer.eps)
        
        # Compute forward activation
        z = x_norm * weight + bias
        
        # Compute stabilized z for relevance propagation
        stabilized_z = z + epsilon * torch.sign(z)
        
        # Compute relevance scaling factor
        scale_factor = relevance / stabilized_z
        
        # Compute input-level relevance
        input_relevance = scale_factor * weight * (1 / torch.sqrt(running_var + layer.eps))
        
        # Scale by input contribution
        input_relevance = input_relevance * (input_tensor - running_mean)
        
        return input_relevance

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
        
        # Ensure proper scaling
        mask = torch.nn.functional.interpolate(mask, size=input_tensor.shape[2:], mode='nearest')
        
        # Distribute relevance evenly (avoiding division by zero)
        mask_sum = mask.sum(dim=(2, 3), keepdim=True)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        normalized_mask = mask / mask_sum
        
        # Distribute relevance evenly
        return relevance * normalized_mask

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
