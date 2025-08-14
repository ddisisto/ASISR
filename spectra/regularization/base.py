"""
Base classes for spectral regularization methods.

This module implements the abstract interfaces for spectral regularizers,
enabling plugin-based regularization strategies in the SPECTRA framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch


class SpectralRegularizer(ABC):
    """
    Base class for spectral regularization methods.
    
    This abstract interface enables different spectral regularization
    strategies (fixed sigma, adaptive, multi-scale) to be used
    interchangeably with any SpectralRegularizedModel.
    """
    
    @abstractmethod
    def compute_loss(self, weight_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute regularization loss for given weight matrices.
        
        Args:
            weight_matrices: List of 2D weight tensors to regularize
            
        Returns:
            Scalar tensor representing the regularization loss
            
        Note:
            This is the core method that enforces spectral constraints.
            Implementation depends on the specific regularization strategy.
        """
        pass
    
    @abstractmethod
    def get_targets(self) -> Dict[int, float]:
        """
        Return current target sigma values for each layer.
        
        Returns:
            Dictionary mapping layer indices to target spectral radius values
            
        Note:
            For fixed regularization: constant targets
            For adaptive regularization: may change based on criticality feedback
        """
        pass
    
    def update_targets(self, criticality_metrics: Dict[str, float]) -> None:
        """
        Update regularization targets based on criticality feedback.
        
        Args:
            criticality_metrics: Dictionary of criticality indicators
            
        Note:
            Optional method for adaptive regularization.
            Fixed regularizers can leave this as no-op.
            Adaptive regularizers use this for real-time target adjustment.
        """
        pass