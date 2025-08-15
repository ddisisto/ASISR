"""
Scale-Invariant Optimization for SPECTRA Framework.

This module implements optimization methods that leverage criticality physics
and scale-invariant principles for improved neural network training.

Core components:
- Critical learning rate schedulers
- Spectral-aware optimizers  
- Criticality measurement utilities
"""

from .schedulers import CriticalityAwareLRScheduler, PowerLawScheduler
from .optimizers import SpectralMomentum, CriticalAdam
from .utils import compute_criticality_distance, track_spectral_evolution

__all__ = [
    'CriticalityAwareLRScheduler',
    'PowerLawScheduler', 
    'SpectralMomentum',
    'CriticalAdam',
    'compute_criticality_distance',
    'track_spectral_evolution'
]