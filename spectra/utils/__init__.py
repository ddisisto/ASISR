"""SPECTRA utilities for configuration, reproducibility, and capacity analysis."""

from .config import SPECTRAConfig, load_config
from .seed import SeedManager, set_seed
from .capacity import (
    CapacityAnalysis,
    count_model_parameters,
    analyze_layer_parameters,
    get_optimal_capacity,
    calculate_capacity_ratio_from_model,
    analyze_model_capacity,
    print_capacity_summary,
    suggest_optimal_architecture,
    categorize_capacity,
    CAPACITY_CATEGORIES
)

__all__ = [
    'SPECTRAConfig',
    'load_config', 
    'SeedManager',
    'set_seed',
    # Phase 3: Capacity Analysis Utilities
    'CapacityAnalysis',
    'count_model_parameters',
    'analyze_layer_parameters', 
    'get_optimal_capacity',
    'calculate_capacity_ratio_from_model',
    'analyze_model_capacity',
    'print_capacity_summary',
    'suggest_optimal_architecture', 
    'categorize_capacity',
    'CAPACITY_CATEGORIES'
]