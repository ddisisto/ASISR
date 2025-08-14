"""ASISR spectral regularization methods."""

from .base import SpectralRegularizer
from .fixed import FixedSpectralRegularizer, create_edge_of_chaos_regularizer

__all__ = [
    'SpectralRegularizer',
    'FixedSpectralRegularizer',
    'create_edge_of_chaos_regularizer'
]