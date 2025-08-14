"""ASISR model architectures with spectral regularization support."""

from .base import SpectralRegularizedModel
from .mlp import SpectralMLP, MLPClassifier, create_boundary_mlp

__all__ = [
    'SpectralRegularizedModel',
    'SpectralMLP', 
    'MLPClassifier',
    'create_boundary_mlp'
]