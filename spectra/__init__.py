"""
SPECTRA: Spectral Performance Control Through Regularization Analysis

A research framework for understanding performance-variance trade-offs in neural 
networks through spectral radius control and dynamic scheduling strategies.
"""

__version__ = "0.2.0"
__author__ = "SPECTRA Research Project"

# Core package imports for convenient access
from .models import SpectralMLP, SpectralRegularizedModel
from .regularization import (
    FixedSpectralRegularizer, 
    DynamicSpectralRegularizer,
    LinearScheduleRegularizer,
    ExponentialScheduleRegularizer
)
from .training import SPECTRAExperiment
from .metrics import CriticalityMonitor
from .utils import SPECTRAConfig, load_config, set_seed
from .optimization import (
    CriticalityAwareLRScheduler,
    PowerLawScheduler,
    compute_criticality_distance,
    track_spectral_evolution
)

__all__ = [
    'SpectralMLP',
    'SpectralRegularizedModel', 
    'FixedSpectralRegularizer',
    'DynamicSpectralRegularizer',
    'LinearScheduleRegularizer',
    'ExponentialScheduleRegularizer',
    'SPECTRAExperiment',
    'CriticalityMonitor',
    'SPECTRAConfig',
    'load_config',
    'set_seed',
    'CriticalityAwareLRScheduler',
    'PowerLawScheduler',
    'compute_criticality_distance',
    'track_spectral_evolution'
]