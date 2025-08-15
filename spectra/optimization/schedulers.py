"""
Criticality-aware learning rate schedulers for scale-invariant optimization.

Implements learning rate scheduling based on spectral properties and 
distance from criticality, enabling physics-principled optimization.
"""

import math
from typing import Optional, Callable
import torch
from torch.optim.lr_scheduler import _LRScheduler
from ..models.base import SpectralRegularizedModel
from .utils import compute_criticality_distance


class CriticalityAwareLRScheduler(_LRScheduler):
    """
    Learning rate scheduler that adapts based on distance from criticality.
    
    Implements scale-invariant learning rate scheduling where the learning rate
    scales with the distance from the critical point (σ ≈ 1.0).
    
    Formula: lr = base_lr * f(|σ - σ_target|)
    where f is a scaling function (power-law by default).
    """
    
    def __init__(self, 
                 optimizer: torch.optim.Optimizer,
                 model: SpectralRegularizedModel,
                 scaling_function: str = 'power_law',
                 alpha: float = 0.5,
                 target_sigma: float = 1.0,
                 min_lr_factor: float = 0.1,
                 max_lr_factor: float = 10.0,
                 last_epoch: int = -1):
        """
        Initialize criticality-aware learning rate scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            model: Model to monitor for spectral properties
            scaling_function: Type of scaling ('power_law', 'exponential', 'linear')
            alpha: Scaling exponent for power-law: lr ∝ distance^(-alpha)
            target_sigma: Target spectral radius (typically 1.0)
            min_lr_factor: Minimum learning rate factor (prevents lr → 0)
            max_lr_factor: Maximum learning rate factor (prevents lr → ∞)
            last_epoch: Last epoch number
            
        Note:
            Power-law scaling implements: lr = base_lr * (distance + ε)^(-alpha)
            where distance = |σ_avg - target_sigma| and ε prevents division by zero.
        """
        self.model = model
        self.scaling_function = scaling_function
        self.alpha = alpha
        self.target_sigma = target_sigma
        self.min_lr_factor = min_lr_factor
        self.max_lr_factor = max_lr_factor
        self.epsilon = 1e-6  # Prevents division by zero
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute current learning rates based on criticality distance."""
        # Compute distance from criticality
        distance = compute_criticality_distance(self.model, self.target_sigma)
        
        # Apply scaling function
        if self.scaling_function == 'power_law':
            scale_factor = (distance + self.epsilon) ** (-self.alpha)
        elif self.scaling_function == 'exponential':
            scale_factor = math.exp(-self.alpha * distance)
        elif self.scaling_function == 'linear':
            scale_factor = max(0.0, 1.0 - self.alpha * distance)
        else:
            raise ValueError(f"Unknown scaling function: {self.scaling_function}")
        
        # Clamp scale factor to prevent extreme values
        scale_factor = max(self.min_lr_factor, min(self.max_lr_factor, scale_factor))
        
        # Apply to all parameter groups
        return [base_lr * scale_factor for base_lr in self.base_lrs]


class PowerLawScheduler(CriticalityAwareLRScheduler):
    """
    Specialized power-law scaling scheduler.
    
    Convenience class for power-law criticality scaling:
    lr(t) = lr_base * |σ(t) - 1.0|^(-α)
    
    This is the core hypothesis for scale-invariant optimization.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 model: SpectralRegularizedModel, 
                 alpha: float = 0.5,
                 target_sigma: float = 1.0,
                 min_lr_factor: float = 0.1,
                 max_lr_factor: float = 10.0,
                 last_epoch: int = -1):
        """
        Initialize power-law criticality scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            model: Model to monitor
            alpha: Power-law exponent (0.5 is a reasonable starting point)
            target_sigma: Target spectral radius
            min_lr_factor: Minimum lr scaling factor
            max_lr_factor: Maximum lr scaling factor
            last_epoch: Last epoch number
            
        Note:
            alpha = 0.5 gives lr ∝ 1/√distance (conservative scaling)
            alpha = 1.0 gives lr ∝ 1/distance (stronger scaling)
            alpha = 2.0 gives lr ∝ 1/distance² (aggressive scaling)
        """
        super().__init__(
            optimizer=optimizer,
            model=model,
            scaling_function='power_law',
            alpha=alpha,
            target_sigma=target_sigma,
            min_lr_factor=min_lr_factor,
            max_lr_factor=max_lr_factor,
            last_epoch=last_epoch
        )


class AdaptiveCriticalityScheduler(_LRScheduler):
    """
    Learning rate scheduler that adapts both to criticality and training progress.
    
    Combines criticality-aware scaling with traditional scheduling (e.g., cosine).
    Provides fallback behavior when criticality information is unreliable.
    """
    
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 model: SpectralRegularizedModel,
                 total_epochs: int,
                 alpha: float = 0.5,
                 target_sigma: float = 1.0,
                 criticality_weight: float = 0.7,
                 cosine_weight: float = 0.3,
                 min_lr_factor: float = 0.01,
                 last_epoch: int = -1):
        """
        Initialize adaptive criticality scheduler.
        
        Args:
            optimizer: Optimizer to schedule
            model: Model to monitor
            total_epochs: Total training epochs
            alpha: Power-law exponent for criticality scaling
            target_sigma: Target spectral radius
            criticality_weight: Weight for criticality component
            cosine_weight: Weight for cosine annealing component
            min_lr_factor: Minimum lr factor (relative to base)
            last_epoch: Last epoch number
            
        Note:
            Final lr = criticality_weight * critical_lr + cosine_weight * cosine_lr
            This provides stable learning even when criticality measurements are noisy.
        """
        self.model = model
        self.total_epochs = total_epochs
        self.alpha = alpha
        self.target_sigma = target_sigma
        self.criticality_weight = criticality_weight
        self.cosine_weight = cosine_weight
        self.min_lr_factor = min_lr_factor
        self.epsilon = 1e-6
        
        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Compute adaptive learning rates combining criticality and cosine annealing."""
        # Criticality component
        distance = compute_criticality_distance(self.model, self.target_sigma)
        critical_scale = (distance + self.epsilon) ** (-self.alpha)
        
        # Cosine annealing component
        if self.total_epochs > 0:
            cosine_scale = 0.5 * (1 + math.cos(math.pi * self.last_epoch / self.total_epochs))
        else:
            cosine_scale = 1.0
        
        # Weighted combination
        combined_scale = (self.criticality_weight * critical_scale + 
                         self.cosine_weight * cosine_scale)
        
        # Apply minimum factor
        combined_scale = max(self.min_lr_factor, combined_scale)
        
        return [base_lr * combined_scale for base_lr in self.base_lrs]


def create_critical_scheduler(scheduler_type: str, 
                            optimizer: torch.optim.Optimizer,
                            model: SpectralRegularizedModel,
                            **kwargs) -> _LRScheduler:
    """
    Factory function for creating criticality-aware schedulers.
    
    Args:
        scheduler_type: Type of scheduler ('power_law', 'adaptive', 'criticality_aware')
        optimizer: Optimizer to schedule
        model: Model to monitor
        **kwargs: Additional arguments for specific schedulers
        
    Returns:
        Configured criticality-aware scheduler
        
    Example:
        scheduler = create_critical_scheduler(
            'power_law', 
            optimizer, 
            model, 
            alpha=0.5, 
            target_sigma=1.0
        )
    """
    if scheduler_type == 'power_law':
        return PowerLawScheduler(optimizer, model, **kwargs)
    elif scheduler_type == 'adaptive':
        return AdaptiveCriticalityScheduler(optimizer, model, **kwargs)
    elif scheduler_type == 'criticality_aware':
        return CriticalityAwareLRScheduler(optimizer, model, **kwargs)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class CriticalityTracker:
    """
    Utility class for tracking criticality evolution during training.
    
    Provides logging and analysis of how spectral properties evolve,
    supporting research into self-organized criticality.
    """
    
    def __init__(self, model: SpectralRegularizedModel, log_interval: int = 10):
        """
        Initialize criticality tracker.
        
        Args:
            model: Model to track
            log_interval: Epochs between measurements
        """
        self.model = model
        self.log_interval = log_interval
        self.history = {}
        self.epoch = 0
    
    def update(self, epoch: int):
        """Update criticality tracking for current epoch."""
        self.epoch = epoch
        
        if epoch % self.log_interval == 0:
            from .utils import track_spectral_evolution
            track_spectral_evolution(self.model, self.history)
    
    def get_approach_analysis(self) -> dict:
        """Analyze how model approaches criticality."""
        from .utils import analyze_natural_criticality_approach
        return analyze_natural_criticality_approach(self.history)
    
    def get_current_distance(self) -> float:
        """Get current distance from criticality."""
        return compute_criticality_distance(self.model)