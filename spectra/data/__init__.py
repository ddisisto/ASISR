"""SPECTRA data loading and preprocessing."""

from .map_loader import BaarleMapLoader
from .synthetic import (
    TwoMoonsLoader, 
    CirclesLoader, 
    CheckerboardLoader,
    create_synthetic_loader
)
from .real_datasets import (
    MNISTLoader,
    FashionMNISTLoader, 
    CIFAR10Loader,
    create_real_dataset_loader
)

__all__ = [
    'BaarleMapLoader',
    'TwoMoonsLoader',
    'CirclesLoader', 
    'CheckerboardLoader',
    'create_synthetic_loader',
    'MNISTLoader',
    'FashionMNISTLoader',
    'CIFAR10Loader', 
    'create_real_dataset_loader'
]