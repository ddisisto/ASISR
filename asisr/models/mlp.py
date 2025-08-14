"""
Multi-Layer Perceptron with spectral regularization support.

This module implements MLP networks that support the spectral regularization
framework, providing hooks for weight matrix analysis and preactivation monitoring.

Extracted from prototypes/SAMPLE-CODE-v1.md with interface compliance.
"""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn
from .base import SpectralRegularizedModel


class SpectralMLP(SpectralRegularizedModel):
    """
    Multi-Layer Perceptron with spectral regularization support.
    
    Implements the SpectralRegularizedModel interface to enable plugin-based
    spectral regularization and criticality monitoring. Designed for boundary
    learning tasks with configurable architecture.
    
    Architecture:
        Input → Linear → ReLU → ... → Linear → ReLU → Linear (output)
        
    The final linear layer is typically excluded from spectral regularization
    to preserve task-specific learning dynamics.
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dims: List[int] = None,
                 output_dim: int = 1,
                 activation: str = 'relu'):
        """
        Initialize the spectral MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer widths. Default: [64, 64]
            output_dim: Output dimension
            activation: Activation function ('relu', 'tanh'). Default: 'relu'
            
        Note:
            Default configuration matches Phase 1 experimental setup.
            Input_dim=2 for coordinate inputs, output_dim=1 for binary classification.
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64]
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.activation_name = activation
        
        # Build network layers
        self._build_network()
        
        # Cache linear layers for spectral regularization
        self.linear_layers = [module for module in self.network if isinstance(module, nn.Linear)]
    
    def _build_network(self) -> None:
        """Build the network architecture."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        
        # Hidden layers with activation
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            
            # Add activation
            if self.activation_name.lower() == 'relu':
                layers.append(nn.ReLU())
            elif self.activation_name.lower() == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {self.activation_name}")
        
        # Output layer (no activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)
    
    def get_regularizable_weights(self) -> List[torch.Tensor]:
        """
        Return weight matrices subject to spectral regularization.
        
        Returns:
            List of weight tensors from hidden layers (excludes output layer)
            
        Note:
            Excludes the final layer to preserve task-specific learning.
            Each tensor is the .weight attribute of a Linear layer.
        """
        # Return all linear layers except the last one
        if len(self.linear_layers) <= 1:
            return []  # No hidden layers to regularize
        
        return [layer.weight for layer in self.linear_layers[:-1]]
    
    def forward_with_preactivations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both output and intermediate preactivations.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, preactivations) where:
            - output: Final network output
            - preactivations: List of linear layer outputs before activation
            
        Note:
            Preactivations are used for dead neuron detection and criticality analysis.
            List length equals number of linear layers (including output).
        """
        preactivations = []
        h = x
        
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                # Store preactivation (before any activation function)
                z = layer(h)
                preactivations.append(z)
                h = z
            else:
                # Apply activation function
                h = layer(h)
        
        return h, preactivations
    
    def get_layer_info(self) -> dict:
        """
        Get information about network architecture.
        
        Returns:
            Dictionary with layer information for analysis and debugging
        """
        info = {
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'activation': self.activation_name,
            'total_layers': len(self.linear_layers),
            'regularizable_layers': len(self.get_regularizable_weights()),
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        # Add layer-wise parameter counts
        layer_params = []
        for i, layer in enumerate(self.linear_layers):
            layer_info = {
                'layer_index': i,
                'input_dim': layer.weight.shape[1],
                'output_dim': layer.weight.shape[0],
                'weight_params': layer.weight.numel(),
                'bias_params': layer.bias.numel() if layer.bias is not None else 0,
                'is_regularized': i < len(self.linear_layers) - 1  # All except last
            }
            layer_params.append(layer_info)
        
        info['layer_details'] = layer_params
        return info


class MLPClassifier(SpectralMLP):
    """
    Binary classification variant of SpectralMLP with sigmoid output.
    
    Convenience class for binary classification tasks like boundary learning.
    Applies sigmoid activation to the output for probability interpretation.
    """
    
    def __init__(self, 
                 input_dim: int = 2,
                 hidden_dims: List[int] = None,
                 activation: str = 'relu'):
        """
        Initialize binary classifier.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer widths. Default: [64, 64]
            activation: Activation function for hidden layers
            
        Note:
            Output dimension fixed to 1 for binary classification.
        """
        super().__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=1,
            activation=activation
        )
    
    def forward_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with sigmoid output for probability interpretation.
        
        Args:
            x: Input tensor
            
        Returns:
            Probabilities in range [0, 1] for positive class
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Binary predictions with configurable threshold.
        
        Args:
            x: Input tensor
            threshold: Classification threshold
            
        Returns:
            Binary predictions (0 or 1)
        """
        probabilities = self.forward_probabilities(x)
        return (probabilities >= threshold).long()


def create_boundary_mlp(hidden_dims: Optional[List[int]] = None,
                       activation: str = 'relu') -> SpectralMLP:
    """
    Factory function for boundary learning MLPs.
    
    Args:
        hidden_dims: Hidden layer dimensions. Default: [64, 64]
        activation: Activation function
        
    Returns:
        SpectralMLP configured for 2D boundary learning
        
    Note:
        Convenience factory matching Phase 1 experimental setup.
        Input dimension fixed to 2 (x, y coordinates).
        Output dimension fixed to 1 (binary classification).
    """
    if hidden_dims is None:
        hidden_dims = [64, 64]
    
    return SpectralMLP(
        input_dim=2,
        hidden_dims=hidden_dims,
        output_dim=1,
        activation=activation
    )