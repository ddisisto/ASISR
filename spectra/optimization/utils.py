"""
Utility functions for criticality-aware optimization.

Provides common functions for measuring and tracking spectral properties
during training, supporting scale-invariant optimization methods.
"""

from typing import List, Dict, Any
import torch
import numpy as np
from ..models.base import SpectralRegularizedModel


def compute_criticality_distance(model: SpectralRegularizedModel, 
                                target_sigma: float = 1.0) -> float:
    """
    Compute distance from criticality for a model.
    
    Args:
        model: SpectralRegularizedModel to analyze
        target_sigma: Target spectral radius (default: 1.0 for edge of chaos)
        
    Returns:
        Average distance from target across all regularizable layers
        
    Note:
        Returns |σ_avg - target_sigma| where σ_avg is mean spectral radius
        across all layers. Used for scale-invariant learning rate computation.
    """
    weights = model.get_regularizable_weights()
    if not weights:
        return 0.0
        
    spectral_radii = []
    for weight in weights:
        sigma = _estimate_spectral_radius(weight)
        spectral_radii.append(sigma)
    
    avg_sigma = np.mean(spectral_radii)
    distance = abs(avg_sigma - target_sigma)
    
    return float(distance)


def track_spectral_evolution(model: SpectralRegularizedModel, 
                           history: Dict[str, List[float]]) -> Dict[str, float]:
    """
    Track spectral radius evolution during training.
    
    Args:
        model: Model to analyze
        history: Dictionary to append spectral measurements
        
    Returns:
        Current spectral measurements
        
    Note:
        Modifies history in-place with new measurements.
        Used for monitoring natural evolution toward criticality.
    """
    weights = model.get_regularizable_weights()
    
    if not weights:
        current_measurements = {
            'avg_spectral_radius': 0.0,
            'max_spectral_radius': 0.0,
            'min_spectral_radius': 0.0,
            'spectral_variance': 0.0
        }
    else:
        spectral_radii = [_estimate_spectral_radius(w) for w in weights]
        
        current_measurements = {
            'avg_spectral_radius': float(np.mean(spectral_radii)),
            'max_spectral_radius': float(np.max(spectral_radii)), 
            'min_spectral_radius': float(np.min(spectral_radii)),
            'spectral_variance': float(np.var(spectral_radii))
        }
    
    # Update history
    for key, value in current_measurements.items():
        if key not in history:
            history[key] = []
        history[key].append(value)
    
    return current_measurements


def compute_layer_wise_criticality(model: SpectralRegularizedModel) -> List[float]:
    """
    Compute criticality distance for each layer individually.
    
    Args:
        model: Model to analyze
        
    Returns:
        List of criticality distances per layer
        
    Note:
        Used for layer-specific optimization and multi-scale analysis.
        Returns distances from σ = 1.0 for each regularizable layer.
    """
    weights = model.get_regularizable_weights()
    layer_criticality = []
    
    for weight in weights:
        sigma = _estimate_spectral_radius(weight)
        distance = abs(sigma - 1.0)
        layer_criticality.append(distance)
    
    return layer_criticality


def _estimate_spectral_radius(weight_matrix: torch.Tensor, n_iters: int = 10) -> float:
    """
    Estimate top singular value using power iteration.
    
    Args:
        weight_matrix: 2D weight tensor
        n_iters: Number of power iteration steps
        
    Returns:
        Estimated spectral radius
        
    Note:
        Efficient implementation from existing SPECTRA framework.
        Maintains gradients for backpropagation compatibility.
    """
    if weight_matrix.dim() != 2:
        return 0.0
    
    W = weight_matrix.detach()
    batch_size, input_dim = W.shape
    
    if input_dim == 0 or batch_size == 0:
        return 0.0
    
    # Initialize random vector
    b = torch.randn(input_dim, device=W.device)
    b = b / (b.norm() + 1e-9)
    
    # Power iteration
    for _ in range(n_iters):
        # Forward: v = W @ b
        v = W @ b
        if v.norm() == 0:
            return 0.0
        
        # Backward: b = W^T @ v, normalize
        b = W.t() @ v
        b = b / (b.norm() + 1e-9)
    
    # Final forward pass to get singular value
    v = W @ b
    spectral_radius = v.norm().item() / (b.norm().item() + 1e-12)
    
    return spectral_radius


def analyze_natural_criticality_approach(spectral_history: Dict[str, List[float]], 
                                       target_sigma: float = 1.0,
                                       tolerance: float = 0.1) -> Dict[str, Any]:
    """
    Analyze how a model naturally approaches criticality during training.
    
    Args:
        spectral_history: History of spectral measurements
        target_sigma: Target spectral radius
        tolerance: Tolerance for "reaching" target
        
    Returns:
        Analysis of criticality approach dynamics
        
    Note:
        Used for self-organized criticality experiments.
        Measures time to reach target, stability, convergence rate.
    """
    if 'avg_spectral_radius' not in spectral_history:
        return {'error': 'No spectral radius history available'}
    
    sigma_history = spectral_history['avg_spectral_radius']
    epochs = len(sigma_history)
    
    # Find when model first reaches near-critical state
    time_to_criticality = None
    for i, sigma in enumerate(sigma_history):
        if abs(sigma - target_sigma) < tolerance:
            time_to_criticality = i
            break
    
    # Analyze stability near critical point
    if time_to_criticality is not None:
        post_critical = sigma_history[time_to_criticality:]
        stability_variance = np.var(post_critical) if post_critical else float('inf')
        
        # Check if it stays near critical
        staying_critical = all(abs(s - target_sigma) < tolerance for s in post_critical)
    else:
        stability_variance = float('inf')
        staying_critical = False
    
    # Compute convergence rate (slope toward target)
    if epochs > 10:
        recent_history = sigma_history[-10:]
        time_points = np.arange(len(recent_history))
        convergence_slope = np.polyfit(time_points, recent_history, 1)[0] if len(recent_history) > 1 else 0.0
    else:
        convergence_slope = 0.0
    
    # Final distance from target
    final_distance = abs(sigma_history[-1] - target_sigma) if sigma_history else float('inf')
    
    return {
        'time_to_criticality': time_to_criticality,
        'reached_criticality': time_to_criticality is not None,
        'stability_variance': float(stability_variance),
        'staying_critical': staying_critical,
        'convergence_slope': float(convergence_slope),
        'final_distance': float(final_distance),
        'epochs_analyzed': epochs
    }