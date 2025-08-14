"""
Dynamic Spectral Regularization - Phase 2B Implementation.

Training-phase-dependent spectral control strategies that optimize performance-variance
trade-offs through adaptive σ scheduling during training.
"""

import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Callable
import torch
import numpy as np

from .base import SpectralRegularizer


class DynamicSpectralRegularizer(SpectralRegularizer):
    """
    Base class for dynamic spectral regularization with training-phase scheduling.
    
    Implements training-phase-dependent σ control where spectral targets evolve
    during training to optimize exploration-exploitation trade-offs.
    """
    
    def __init__(self, initial_sigma: float, final_sigma: float, 
                 total_epochs: int, regularization_strength: float = 0.1,
                 warmup_epochs: int = 0):
        """
        Initialize dynamic spectral regularizer.
        
        Args:
            initial_sigma: Starting σ value (typically higher for exploration)
            final_sigma: Target σ value (typically lower for convergence)
            total_epochs: Total training epochs for schedule computation
            regularization_strength: Strength of spectral regularization
            warmup_epochs: Number of epochs before scheduling begins
        """
        self.initial_sigma = initial_sigma
        self.final_sigma = final_sigma
        self.total_epochs = total_epochs
        self.regularization_strength = regularization_strength
        self.warmup_epochs = warmup_epochs
        
        self.current_epoch = 0
        self.current_sigma = initial_sigma
        
    @abstractmethod
    def compute_sigma_schedule(self, epoch: int) -> float:
        """
        Compute current σ value based on training epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            Current σ target value
        """
        pass
    
    def update_epoch(self, epoch: int) -> None:
        """Update current epoch and recompute σ target."""
        self.current_epoch = epoch
        self.current_sigma = self.compute_sigma_schedule(epoch)
    
    def compute_loss(self, weight_matrices: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute spectral regularization loss with current dynamic σ target.
        
        Args:
            weight_matrices: List of weight tensors to regularize
            
        Returns:
            Spectral regularization loss tensor
        """
        if not weight_matrices:
            return torch.tensor(0.0, requires_grad=True)
        
        total_loss = torch.tensor(0.0, device=weight_matrices[0].device, requires_grad=True)
        
        for weight in weight_matrices:
            if weight.requires_grad and weight.numel() > 1:
                # Compute largest singular value via power iteration
                u, v = self._power_iteration(weight)
                spectral_radius = torch.sqrt(torch.sum((weight @ v) ** 2))
                
                # Dynamic σ targeting loss
                sigma_loss = (spectral_radius - self.current_sigma) ** 2
                total_loss = total_loss + sigma_loss
        
        return self.regularization_strength * total_loss
    
    def get_targets(self) -> Dict[int, float]:
        """Return current σ target for all layers."""
        return {0: self.current_sigma}  # Single target for all layers
    
    def _power_iteration(self, matrix: torch.Tensor, 
                        num_iterations: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dominant singular vectors via power iteration.
        
        Args:
            matrix: Input weight matrix
            num_iterations: Number of power iteration steps
            
        Returns:
            Tuple of (left_singular_vector, right_singular_vector)
        """
        # Initialize random vectors
        m, n = matrix.shape
        u = torch.randn(m, device=matrix.device, dtype=matrix.dtype)
        v = torch.randn(n, device=matrix.device, dtype=matrix.dtype)
        
        u = u / torch.norm(u)
        v = v / torch.norm(v)
        
        for _ in range(num_iterations):
            # Power iteration: v = A^T u, u = A v
            v = matrix.T @ u
            v = v / torch.norm(v)
            u = matrix @ v
            u = u / torch.norm(u)
        
        return u, v


class LinearScheduleRegularizer(DynamicSpectralRegularizer):
    """
    Linear σ scheduling: σ(t) = σ_initial + (σ_final - σ_initial) * t/T
    
    Simple linear interpolation between initial and final σ values.
    """
    
    def compute_sigma_schedule(self, epoch: int) -> float:
        """Compute σ value using linear schedule."""
        if epoch < self.warmup_epochs:
            return self.initial_sigma
        
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        
        if adjusted_total <= 0:
            return self.final_sigma
        
        progress = min(adjusted_epoch / adjusted_total, 1.0)
        return self.initial_sigma + (self.final_sigma - self.initial_sigma) * progress


class ExponentialScheduleRegularizer(DynamicSpectralRegularizer):
    """
    Exponential σ scheduling: σ(t) = σ_final + (σ_initial - σ_final) * exp(-γt)
    
    Exponential decay from initial to final σ with configurable decay rate.
    """
    
    def __init__(self, initial_sigma: float, final_sigma: float, 
                 total_epochs: int, decay_rate: float = 5.0,
                 regularization_strength: float = 0.1, warmup_epochs: int = 0):
        """
        Initialize exponential schedule regularizer.
        
        Args:
            initial_sigma: Starting σ value
            final_sigma: Target σ value
            total_epochs: Total training epochs
            decay_rate: Controls exponential decay speed (higher = faster decay)
            regularization_strength: Strength of spectral regularization
            warmup_epochs: Number of epochs before scheduling begins
        """
        super().__init__(initial_sigma, final_sigma, total_epochs, 
                        regularization_strength, warmup_epochs)
        self.decay_rate = decay_rate
    
    def compute_sigma_schedule(self, epoch: int) -> float:
        """Compute σ value using exponential decay schedule."""
        if epoch < self.warmup_epochs:
            return self.initial_sigma
        
        adjusted_epoch = epoch - self.warmup_epochs
        adjusted_total = self.total_epochs - self.warmup_epochs
        
        if adjusted_total <= 0:
            return self.final_sigma
        
        # Normalized progress [0, 1]
        progress = min(adjusted_epoch / adjusted_total, 1.0)
        
        # Exponential decay: exp(-γ * progress)
        decay_factor = math.exp(-self.decay_rate * progress)
        
        return self.final_sigma + (self.initial_sigma - self.final_sigma) * decay_factor


class StepScheduleRegularizer(DynamicSpectralRegularizer):
    """
    Step-wise σ scheduling: σ changes in discrete steps at specified milestones.
    
    Useful for phase-based training where distinct exploration/exploitation phases
    are desired.
    """
    
    def __init__(self, sigma_schedule: List[tuple[int, float]], 
                 total_epochs: int, regularization_strength: float = 0.1):
        """
        Initialize step schedule regularizer.
        
        Args:
            sigma_schedule: List of (epoch, sigma) pairs defining step schedule
            total_epochs: Total training epochs
            regularization_strength: Strength of spectral regularization
        """
        # Extract initial and final sigma from schedule
        if not sigma_schedule:
            raise ValueError("sigma_schedule cannot be empty")
        
        sorted_schedule = sorted(sigma_schedule, key=lambda x: x[0])
        initial_sigma = sorted_schedule[0][1]
        final_sigma = sorted_schedule[-1][1]
        
        super().__init__(initial_sigma, final_sigma, total_epochs, 
                        regularization_strength, warmup_epochs=0)
        
        self.sigma_schedule = sorted_schedule
    
    def compute_sigma_schedule(self, epoch: int) -> float:
        """Compute σ value using step schedule."""
        # Find the appropriate step
        current_sigma = self.sigma_schedule[0][1]  # Default to first value
        
        for step_epoch, step_sigma in self.sigma_schedule:
            if epoch >= step_epoch:
                current_sigma = step_sigma
            else:
                break
        
        return current_sigma


class AdaptiveScheduleRegularizer(DynamicSpectralRegularizer):
    """
    Adaptive σ scheduling based on training metrics feedback.
    
    Adjusts σ values in response to criticality indicators, loss plateaus,
    or other training signals for data-driven optimization.
    """
    
    def __init__(self, initial_sigma: float, final_sigma: float,
                 total_epochs: int, adaptation_rate: float = 0.1,
                 regularization_strength: float = 0.1, 
                 warmup_epochs: int = 10):
        """
        Initialize adaptive schedule regularizer.
        
        Args:
            initial_sigma: Starting σ value
            final_sigma: Target σ value  
            total_epochs: Total training epochs
            adaptation_rate: Rate of adaptation to training signals
            regularization_strength: Strength of spectral regularization
            warmup_epochs: Number of epochs before adaptation begins
        """
        super().__init__(initial_sigma, final_sigma, total_epochs,
                        regularization_strength, warmup_epochs)
        self.adaptation_rate = adaptation_rate
        
        # Training history for adaptation
        self.loss_history: List[float] = []
        self.criticality_history: List[float] = []
        self.adaptation_target = initial_sigma
    
    def update_training_metrics(self, loss: float, 
                               criticality_score: Optional[float] = None) -> None:
        """
        Update training metrics for adaptive scheduling.
        
        Args:
            loss: Current training loss
            criticality_score: Current criticality score (optional)
        """
        self.loss_history.append(loss)
        if criticality_score is not None:
            self.criticality_history.append(criticality_score)
        
        # Adaptive logic: increase σ if loss is plateauing (need exploration)
        if len(self.loss_history) >= 5:
            recent_losses = self.loss_history[-5:]
            loss_variance = np.var(recent_losses)
            
            if loss_variance < 1e-6:  # Loss plateau detected
                # Increase σ slightly to encourage exploration
                self.adaptation_target = min(
                    self.adaptation_target * (1 + self.adaptation_rate),
                    self.initial_sigma
                )
            else:
                # Decrease σ toward final target for convergence
                self.adaptation_target = max(
                    self.adaptation_target * (1 - self.adaptation_rate * 0.5),
                    self.final_sigma
                )
    
    def compute_sigma_schedule(self, epoch: int) -> float:
        """Compute σ value using adaptive schedule."""
        if epoch < self.warmup_epochs:
            return self.initial_sigma
        
        # Use adaptation target if available, otherwise fall back to linear
        if hasattr(self, 'adaptation_target'):
            return self.adaptation_target
        else:
            # Fallback to linear schedule
            adjusted_epoch = epoch - self.warmup_epochs
            adjusted_total = self.total_epochs - self.warmup_epochs
            progress = min(adjusted_epoch / adjusted_total, 1.0) if adjusted_total > 0 else 1.0
            return self.initial_sigma + (self.final_sigma - self.initial_sigma) * progress


# Factory function for easy instantiation
def create_dynamic_regularizer(schedule_type: str, **kwargs) -> DynamicSpectralRegularizer:
    """
    Factory function to create dynamic spectral regularizers.
    
    Args:
        schedule_type: Type of scheduling ('linear', 'exponential', 'step', 'adaptive')
        **kwargs: Arguments for specific regularizer types
        
    Returns:
        Configured dynamic spectral regularizer
    """
    if schedule_type == 'linear':
        return LinearScheduleRegularizer(**kwargs)
    elif schedule_type == 'exponential':
        return ExponentialScheduleRegularizer(**kwargs)
    elif schedule_type == 'step':
        return StepScheduleRegularizer(**kwargs)
    elif schedule_type == 'adaptive':
        return AdaptiveScheduleRegularizer(**kwargs)
    else:
        raise ValueError(f"Unknown schedule type: {schedule_type}")