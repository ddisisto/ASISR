"""SPECTRA data loading and preprocessing."""

from .map_loader import BaarleMapLoader
from .synthetic import (
    TwoMoonsLoader, 
    CirclesLoader, 
    CheckerboardLoader,
    create_synthetic_loader
)

__all__ = [
    'BaarleMapLoader',
    'TwoMoonsLoader',
    'CirclesLoader', 
    'CheckerboardLoader',
    'create_synthetic_loader'
]