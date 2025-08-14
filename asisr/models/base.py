"""
Base classes for spectral regularized models.

This module implements the abstract interfaces defined in ARCHITECTURE.md,
providing the plugin architecture foundation for the ASISR project.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import torch
import torch.nn as nn


class SpectralRegularizedModel(nn.Module, ABC):
    """
    Base class for neural network models supporting spectral regularization.
    
    This abstract interface enables plugin-based spectral regularization
    by exposing weight matrices and preactivations for analysis.
    
    All ASISR models must implement this interface to work with the
    regularization and criticality monitoring systems.
    """
    
    @abstractmethod
    def get_regularizable_weights(self) -> List[torch.Tensor]:
        """
        Return weight matrices subject to spectral regularization.
        
        Returns:
            List of weight tensors (typically Linear layer weights)
            that should have their spectral radius controlled.
            
        Note:
            Usually excludes output layer weights to preserve task learning.
            Each tensor should be 2D (weight matrices).
        """
        pass
    
    @abstractmethod
    def forward_with_preactivations(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass returning both output and intermediate preactivations.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (output, preactivations) where:
            - output: Final model output
            - preactivations: List of pre-activation tensors for criticality analysis
            
        Note:
            Preactivations are the linear layer outputs before activation functions.
            Used for dead neuron detection and perturbation sensitivity analysis.
        """
        pass
    
    def spectral_loss(self, regularizer) -> torch.Tensor:
        """
        Compute spectral regularization loss using provided regularizer.
        
        Args:
            regularizer: SpectralRegularizer instance
            
        Returns:
            Scalar tensor representing regularization loss
            
        Note:
            This method provides the standard interface between models
            and regularizers, enabling plugin-based regularization.
        """
        return regularizer.compute_loss(self.get_regularizable_weights())