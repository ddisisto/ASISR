"""SPECTRA spectral regularization methods."""

from .base import SpectralRegularizer
from .fixed import FixedSpectralRegularizer, create_edge_of_chaos_regularizer
from .dynamic import (
    DynamicSpectralRegularizer,
    LinearScheduleRegularizer,
    ExponentialScheduleRegularizer,
    StepScheduleRegularizer,
    AdaptiveScheduleRegularizer,
    create_dynamic_regularizer
)
from .adaptive import (
    CapacityAdaptiveSpectralRegularizer,
    calculate_capacity_ratio,
    create_capacity_adaptive_regularizer,
    CriticalityAdaptiveRegularizer
)

__all__ = [
    'SpectralRegularizer',
    'FixedSpectralRegularizer',
    'create_edge_of_chaos_regularizer',
    'DynamicSpectralRegularizer',
    'LinearScheduleRegularizer',
    'ExponentialScheduleRegularizer', 
    'StepScheduleRegularizer',
    'AdaptiveScheduleRegularizer',
    'create_dynamic_regularizer',
    # Phase 3: Capacity-Adaptive Regularization
    'CapacityAdaptiveSpectralRegularizer',
    'calculate_capacity_ratio', 
    'create_capacity_adaptive_regularizer',
    'CriticalityAdaptiveRegularizer'
]