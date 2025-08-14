"""
Fixed spectral regularization targeting specific sigma values.

This module implements fixed spectral radius targeting, the foundation
of the ASISR approach. Enforces spectral radius σ ≈ 1.0 to maintain
networks at the "edge of chaos".

Extracted and refined from prototypes/SAMPLE-CODE-v1.md.
"""

from typing import List, Dict, Optional
import torch
import numpy as np
from .base import SpectralRegularizer


class FixedSpectralRegularizer(SpectralRegularizer):
    """
    Fixed spectral regularization targeting specific sigma values.
    
    Enforces spectral radius constraints on weight matrices using
    L2 penalty on deviation from target values. Core implementation
    of the ASISR hypothesis that σ ≈ 1.0 enables optimal learning.
    
    Loss Formula:
        L_spectral = λ * Σ_i (σ_i - σ_target)²
        
    Where σ_i is the top singular value of layer i weight matrix.
    """
    
    def __init__(self, 
                 target_sigma: float = 1.0,
                 regularization_strength: float = 0.1,
                 power_iterations: int = 10):
        """
        Initialize fixed spectral regularizer.
        
        Args:
            target_sigma: Target spectral radius (typically 1.0 for edge of chaos)
            regularization_strength: Regularization weight λ in loss function
            power_iterations: Number of power iterations for singular value estimation
            
        Note:
            Default target_sigma=1.0 based on ASISR hypothesis.
            Power iterations provide efficient spectral radius estimation.
        """
        self.target_sigma = target_sigma
        self.regularization_strength = regularization_strength
        self.power_iterations = power_iterations
        
        # Validate parameters
        if target_sigma <= 0:
            raise ValueError(f"target_sigma must be positive, got {target_sigma}")
        if regularization_strength < 0:
            raise ValueError(f"regularization_strength must be non-negative, got {regularization_strength}")
        if power_iterations < 1:
            raise ValueError(f"power_iterations must be at least 1, got {power_iterations}")
    
    def compute_loss(self, weight_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute spectral regularization loss for given weight matrices.
        
        Args:
            weight_matrices: List of 2D weight tensors to regularize
            
        Returns:
            Scalar tensor representing regularization loss
            
        Note:
            Returns zero tensor if no weights provided (graceful handling).
            Uses efficient power iteration for singular value estimation.
        """
        if not weight_matrices:
            # No weights to regularize - return zero loss
            return torch.tensor(0.0, requires_grad=True)
        
        device = weight_matrices[0].device
        total_loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        for weight_matrix in weight_matrices:
            # Estimate top singular value
            sigma = self._estimate_spectral_radius(weight_matrix)
            
            # L2 penalty on deviation from target
            deviation = sigma - self.target_sigma
            total_loss = total_loss + deviation ** 2
        
        # Apply regularization strength
        regularization_loss = self.regularization_strength * total_loss
        
        return regularization_loss
    
    def get_targets(self) -> Dict[int, float]:
        """
        Return current target sigma values for each layer.
        
        Returns:
            Dictionary mapping layer indices to target spectral radius values
            
        Note:
            For fixed regularization, all layers have the same target.
            Layer indices are implicit (0, 1, 2, ...).
        """
        # Fixed regularization uses the same target for all layers
        # We don't know the number of layers here, so return a callable
        # In practice, this is used for monitoring/logging
        return {'default': self.target_sigma}
    
    def _estimate_spectral_radius(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """
        Estimate top singular value using power iteration method.
        
        Args:
            weight_matrix: 2D weight tensor
            
        Returns:
            Scalar tensor with estimated spectral radius
            
        Note:
            Efficient implementation from prototypes/SAMPLE-CODE-v1.md.
            Maintains gradients for backpropagation.
        """
        if weight_matrix.dim() != 2:
            raise ValueError(f"Expected 2D weight matrix, got shape {weight_matrix.shape}")
        
        W = weight_matrix
        batch_size, input_dim = W.shape
        
        if input_dim == 0 or batch_size == 0:
            return torch.tensor(0.0, device=W.device, requires_grad=True)
        
        # Initialize random vector on same device
        b = torch.randn(input_dim, device=W.device, requires_grad=False)
        b = b / (b.norm() + 1e-9)
        
        # Power iteration
        for _ in range(self.power_iterations):
            # Forward: v = W @ b
            v = W @ b
            
            if v.norm() == 0:
                return torch.tensor(0.0, device=W.device, requires_grad=True)
            
            # Backward: b = W^T @ v, normalize
            b = W.t() @ v
            b_norm = b.norm()
            
            if b_norm == 0:
                return torch.tensor(0.0, device=W.device, requires_grad=True)
            
            b = b / (b_norm + 1e-9)
        
        # Final forward pass to get singular value estimate
        v = W @ b
        v_norm = v.norm()
        b_norm = b.norm()
        
        # Avoid division by zero
        spectral_radius = v_norm / (b_norm + 1e-12)
        
        return spectral_radius
    
    def get_current_sigmas(self, weight_matrices: List[torch.Tensor]) -> List[float]:
        """
        Get current spectral radii for monitoring purposes.
        
        Args:
            weight_matrices: List of weight tensors
            
        Returns:
            List of current spectral radius values (detached from computation graph)
            
        Note:
            Utility method for logging and monitoring during training.
            Values are detached to avoid interfering with gradients.
        """
        sigmas = []
        
        with torch.no_grad():
            for weight_matrix in weight_matrices:
                sigma = self._estimate_spectral_radius(weight_matrix)
                sigmas.append(sigma.item())
        
        return sigmas
    
    def get_config(self) -> Dict[str, float]:
        """
        Get regularizer configuration.
        
        Returns:
            Dictionary with regularizer parameters
        """
        return {
            'target_sigma': self.target_sigma,
            'regularization_strength': self.regularization_strength,
            'power_iterations': self.power_iterations
        }
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"FixedSpectralRegularizer("
                f"target_sigma={self.target_sigma}, "
                f"strength={self.regularization_strength}, "
                f"power_iters={self.power_iterations})")


def create_edge_of_chaos_regularizer(regularization_strength: float = 0.1) -> FixedSpectralRegularizer:
    """
    Factory function for edge-of-chaos spectral regularization.
    
    Args:
        regularization_strength: Regularization weight in loss function
        
    Returns:
        FixedSpectralRegularizer configured for σ = 1.0 targeting
        
    Note:
        Convenience factory for the core ASISR hypothesis.
        Uses σ = 1.0 as the edge-of-chaos target.
    """
    return FixedSpectralRegularizer(
        target_sigma=1.0,
        regularization_strength=regularization_strength,
        power_iterations=10
    )