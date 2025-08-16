"""
Adaptive Spectral Regularization - Phase 3 Implementation.

Capacity-aware spectral control that adapts σ trajectories based on network
capacity relative to problem complexity, enabling universal optimization improvements.
"""

import math
from typing import Dict, List, Optional, Union
import torch
import numpy as np

from .base import SpectralRegularizer
from .dynamic import DynamicSpectralRegularizer


class CapacityAdaptiveSpectralRegularizer(DynamicSpectralRegularizer):
    """
    Capacity-adaptive spectral regularization based on Phase 2D breakthrough discoveries.
    
    Implements the core Phase 3 innovation: σ_initial = σ_base * (capacity_ratio)^β
    where capacity_ratio represents network parameters relative to optimal capacity
    for the given problem complexity.
    
    This addresses the fundamental finding that:
    - Under-parameterized networks (8x8) are hurt by standard linear scheduling
    - Optimally parameterized networks (16x16-32x32) benefit significantly  
    - Over-parameterized networks (64x64+) show diminishing benefits
    """
    
    def __init__(self, 
                 capacity_ratio: float,
                 beta: float = -0.2,
                 sigma_base: float = 2.5,
                 final_sigma: float = 1.0,
                 total_epochs: int = 100,
                 regularization_strength: float = 0.1,
                 warmup_epochs: int = 0):
        """
        Initialize capacity-adaptive spectral regularizer.
        
        Args:
            capacity_ratio: Network parameter count / optimal parameter count
            beta: Capacity scaling exponent (negative values give more exploration to small networks)
            sigma_base: Base σ value for capacity scaling (Phase 2B validated value)
            final_sigma: Target final σ value (typically 1.0 for criticality)
            total_epochs: Total training epochs for schedule computation
            regularization_strength: Strength of spectral regularization loss
            warmup_epochs: Number of epochs before scheduling begins
            
        Note:
            Key insight: β < 0 gives under-parameterized networks higher initial σ
            for more exploration time, while over-parameterized networks get lower
            initial σ since they need less regularization.
        """
        self.capacity_ratio = capacity_ratio
        self.beta = beta
        self.sigma_base = sigma_base
        
        # Compute capacity-adaptive initial sigma
        initial_sigma = self._compute_capacity_adaptive_sigma(capacity_ratio, beta, sigma_base)
        
        # Initialize parent with computed initial sigma
        super().__init__(
            initial_sigma=initial_sigma,
            final_sigma=final_sigma,
            total_epochs=total_epochs,
            regularization_strength=regularization_strength,
            warmup_epochs=warmup_epochs
        )
        
        # Store capacity parameters for debugging/analysis
        self.computed_initial_sigma = initial_sigma
        
    def _compute_capacity_adaptive_sigma(self, capacity_ratio: float, 
                                       beta: float, sigma_base: float) -> float:
        """
        Compute initial σ value based on capacity ratio and scaling parameter.
        
        Core Phase 3 formula: σ_initial = σ_base * (capacity_ratio)^β
        
        Args:
            capacity_ratio: Network parameters / optimal parameters for problem
            beta: Scaling exponent controlling capacity dependence
            sigma_base: Base σ value for scaling
            
        Returns:
            Capacity-adapted initial σ value
            
        Note:
            This is the mathematical heart of the Phase 3 breakthrough.
            Different β values enable different capacity adaptation strategies.
        """
        if capacity_ratio <= 0:
            raise ValueError(f"Capacity ratio must be positive, got {capacity_ratio}")
        
        # Prevent extreme values that could cause numerical issues
        capacity_ratio = max(0.1, min(capacity_ratio, 10.0))
        
        adapted_sigma = sigma_base * (capacity_ratio ** beta)
        
        # Ensure adapted sigma is reasonable (between 0.5 and 5.0)
        adapted_sigma = max(0.5, min(adapted_sigma, 5.0))
        
        return adapted_sigma
    
    def compute_sigma_schedule(self, epoch: int) -> float:
        """
        Compute current σ value using linear decay from capacity-adapted initial value.
        
        Uses the same linear scheduling proven in Phase 2B, but with capacity-adaptive
        initial σ value computed based on network-problem matching.
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            Current σ target value for this epoch
        """
        if epoch < self.warmup_epochs:
            return self.initial_sigma
        
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        
        if adjusted_total <= 0:
            return self.final_sigma
        
        # Linear decay from capacity-adapted initial to final sigma
        progress = min(adjusted_epoch / adjusted_total, 1.0)
        return self.initial_sigma + (self.final_sigma - self.initial_sigma) * progress
    
    def get_capacity_info(self) -> Dict[str, float]:
        """
        Return capacity adaptation parameters for analysis and debugging.
        
        Returns:
            Dictionary containing capacity ratio, beta, and computed initial sigma
        """
        return {
            'capacity_ratio': self.capacity_ratio,
            'beta': self.beta,
            'sigma_base': self.sigma_base,
            'computed_initial_sigma': self.computed_initial_sigma,
            'final_sigma': self.final_sigma
        }


def calculate_capacity_ratio(model_params: int, dataset_name: str) -> float:
    """
    Calculate capacity ratio based on empirical Phase 2D optimal architectures.
    
    This function encodes the Phase 2D breakthrough findings about optimal
    network capacity for different problem complexities.
    
    Args:
        model_params: Total number of trainable parameters in the model
        dataset_name: Name of dataset to determine optimal capacity
        
    Returns:
        Capacity ratio (actual_params / optimal_params)
        
    Note:
        Optimal parameter counts derived from Phase 2D architecture sensitivity:
        - TwoMoons: 16x16 (464 params) shows peak effectiveness
        - Circles: 32x32 (1,664 params) predicted optimal  
        - Belgium: 64x64 (5,888 params) predicted optimal
    """
    # Empirically derived optimal parameter counts from Phase 2D
    optimal_params_map = {
        'two_moons': 464,     # 16x16 architecture (Phase 2D validated)
        'twomoons': 464,      # Alternative naming
        'circles': 1664,      # 32x32 architecture (Phase 2D predicted) 
        'belgium': 5888,      # 64x64 architecture (Phase 2D predicted)
        'belgium_netherlands': 5888,  # Full dataset name
    }
    
    # Normalize dataset name for lookup
    dataset_key = dataset_name.lower().replace('-', '_').replace(' ', '_')
    
    if dataset_key not in optimal_params_map:
        # Default to TwoMoons optimal for unknown datasets
        optimal_params = 464
        print(f"Warning: Unknown dataset '{dataset_name}', using TwoMoons optimal ({optimal_params} params)")
    else:
        optimal_params = optimal_params_map[dataset_key]
    
    return model_params / optimal_params


def create_capacity_adaptive_regularizer(
    model_params: int,
    dataset_name: str,
    beta: float = -0.2,
    **kwargs
) -> CapacityAdaptiveSpectralRegularizer:
    """
    Factory function to create capacity-adaptive regularizers with automatic capacity calculation.
    
    Args:
        model_params: Total number of trainable parameters
        dataset_name: Dataset name for optimal capacity lookup
        beta: Capacity scaling exponent 
        **kwargs: Additional arguments for CapacityAdaptiveSpectralRegularizer
        
    Returns:
        Configured capacity-adaptive spectral regularizer
        
    Example:
        >>> # For 8x8 network (~120 params) on TwoMoons
        >>> regularizer = create_capacity_adaptive_regularizer(
        ...     model_params=120, 
        ...     dataset_name='two_moons',
        ...     beta=-0.2,
        ...     total_epochs=50
        ... )
        >>> # Capacity ratio = 120/464 ≈ 0.26
        >>> # Initial sigma = 2.5 * (0.26)^(-0.2) ≈ 2.85 (higher for exploration)
    """
    capacity_ratio = calculate_capacity_ratio(model_params, dataset_name)
    
    return CapacityAdaptiveSpectralRegularizer(
        capacity_ratio=capacity_ratio,
        beta=beta,
        **kwargs
    )


# Backward compatibility with Phase 2 adaptive regularizer
class CriticalityAdaptiveRegularizer(CapacityAdaptiveSpectralRegularizer):
    """
    Alias for CapacityAdaptiveSpectralRegularizer for backward compatibility.
    
    This maintains compatibility with any Phase 2 code that might reference
    the older adaptive regularization concepts.
    """
    pass