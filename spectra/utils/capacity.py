"""
Capacity analysis utilities for Phase 3 adaptive optimization.

This module provides utilities for calculating network capacity, analyzing
parameter distributions, and supporting capacity-complexity matching experiments.
"""

from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class CapacityAnalysis:
    """
    Results of network capacity analysis.
    
    Attributes:
        total_params: Total number of trainable parameters
        layer_params: Parameters per layer
        capacity_ratio: Ratio to optimal capacity for dataset
        architecture_type: Type of architecture (MLP, CNN, etc.)
        optimal_params: Optimal parameter count for dataset
        dataset_name: Dataset used for capacity calculation
    """
    total_params: int
    layer_params: Dict[str, int] 
    capacity_ratio: float
    architecture_type: str
    optimal_params: int
    dataset_name: str


def count_model_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """
    Count total parameters in a PyTorch model.
    
    Args:
        model: PyTorch model to analyze
        trainable_only: Whether to count only trainable parameters
        
    Returns:
        Total number of parameters
        
    Example:
        >>> from spectra.models import SpectralMLP
        >>> model = SpectralMLP([2, 16, 16, 1])
        >>> total_params = count_model_parameters(model)
        >>> print(f"Model has {total_params} parameters")
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def analyze_layer_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Analyze parameter distribution across model layers.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary mapping layer names to parameter counts
        
    Example:
        >>> layer_params = analyze_layer_parameters(model)
        >>> for layer_name, param_count in layer_params.items():
        ...     print(f"{layer_name}: {param_count} parameters")
    """
    layer_params = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer_params[name] = param.numel()
    
    return layer_params


def get_optimal_capacity(dataset_name: str) -> int:
    """
    Get optimal parameter count for dataset based on Phase 2D empirical findings.
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Optimal parameter count for the dataset
        
    Note:
        These values are derived from Phase 2D capacity threshold experiments:
        - TwoMoons: 16x16 architecture (464 params) shows peak effectiveness
        - Circles: 32x32 architecture (1,664 params) predicted optimal
        - Belgium: 64x64 architecture (5,888 params) predicted optimal
    """
    # Import the function from adaptive module to maintain single source of truth
    from ..regularization.adaptive import calculate_capacity_ratio
    
    # Use a reference model with 1 parameter to extract optimal values
    optimal_params_map = {
        'two_moons': 464,
        'twomoons': 464,
        'circles': 1664,
        'belgium': 5888,
        'belgium_netherlands': 5888,
    }
    
    dataset_key = dataset_name.lower().replace('-', '_').replace(' ', '_')
    return optimal_params_map.get(dataset_key, 464)  # Default to TwoMoons


def calculate_capacity_ratio_from_model(model: nn.Module, dataset_name: str) -> float:
    """
    Calculate capacity ratio for a model relative to optimal capacity for dataset.
    
    Args:
        model: PyTorch model to analyze
        dataset_name: Dataset name for optimal capacity lookup
        
    Returns:
        Capacity ratio (model_params / optimal_params)
        
    Example:
        >>> from spectra.models import SpectralMLP
        >>> model = SpectralMLP([2, 8, 8, 1])  # Small 8x8 network
        >>> ratio = calculate_capacity_ratio_from_model(model, 'two_moons')
        >>> print(f"Capacity ratio: {ratio:.3f}")  # Should be ~0.26 for 8x8
    """
    from ..regularization.adaptive import calculate_capacity_ratio
    
    model_params = count_model_parameters(model)
    return calculate_capacity_ratio(model_params, dataset_name)


def analyze_model_capacity(model: nn.Module, dataset_name: str) -> CapacityAnalysis:
    """
    Comprehensive capacity analysis for a model-dataset combination.
    
    Args:
        model: PyTorch model to analyze
        dataset_name: Dataset name for capacity calculation
        
    Returns:
        CapacityAnalysis object with comprehensive capacity information
        
    Example:
        >>> analysis = analyze_model_capacity(model, 'two_moons')
        >>> print(f"Model type: {analysis.architecture_type}")
        >>> print(f"Total params: {analysis.total_params}")
        >>> print(f"Capacity ratio: {analysis.capacity_ratio:.3f}")
        >>> print(f"Optimal for dataset: {analysis.optimal_params}")
    """
    total_params = count_model_parameters(model)
    layer_params = analyze_layer_parameters(model) 
    optimal_params = get_optimal_capacity(dataset_name)
    capacity_ratio = total_params / optimal_params
    
    # Determine architecture type
    architecture_type = type(model).__name__
    if hasattr(model, 'layers'):
        if all(isinstance(layer, nn.Linear) for layer in model.layers if isinstance(layer, nn.Module)):
            architecture_type = "MLP"
    
    return CapacityAnalysis(
        total_params=total_params,
        layer_params=layer_params,
        capacity_ratio=capacity_ratio,
        architecture_type=architecture_type,
        optimal_params=optimal_params,
        dataset_name=dataset_name
    )


def print_capacity_summary(analysis: CapacityAnalysis) -> None:
    """
    Print a formatted summary of capacity analysis.
    
    Args:
        analysis: CapacityAnalysis object to summarize
        
    Example:
        >>> analysis = analyze_model_capacity(model, 'two_moons')
        >>> print_capacity_summary(analysis)
        
        Capacity Analysis Summary:
        ========================
        Architecture: MLP
        Dataset: two_moons
        Total Parameters: 120
        Optimal Parameters: 464
        Capacity Ratio: 0.259
        Capacity Category: Under-parameterized
        Expected Linear Scheduling Effect: Negative (hurt by exploration phase)
    """
    print("Capacity Analysis Summary:")
    print("=" * 25)
    print(f"Architecture: {analysis.architecture_type}")
    print(f"Dataset: {analysis.dataset_name}")
    print(f"Total Parameters: {analysis.total_params:,}")
    print(f"Optimal Parameters: {analysis.optimal_params:,}")
    print(f"Capacity Ratio: {analysis.capacity_ratio:.3f}")
    
    # Capacity categorization based on Phase 2D findings
    if analysis.capacity_ratio < 0.5:
        category = "Under-parameterized"
        effect = "Negative (hurt by exploration phase)"
    elif analysis.capacity_ratio < 2.0:
        category = "Optimal capacity"
        effect = "Positive (benefits from scheduling)"
    else:
        category = "Over-parameterized"
        effect = "Diminishing (regularization less critical)"
    
    print(f"Capacity Category: {category}")
    print(f"Expected Linear Scheduling Effect: {effect}")
    
    # Layer breakdown
    if len(analysis.layer_params) <= 10:  # Only show if reasonable number of layers
        print("\nLayer Parameter Distribution:")
        for layer_name, param_count in analysis.layer_params.items():
            print(f"  {layer_name}: {param_count:,} parameters")


def suggest_optimal_architecture(dataset_name: str, 
                                architecture_type: str = "MLP") -> Tuple[List[int], str]:
    """
    Suggest optimal architecture dimensions based on Phase 2D capacity findings.
    
    Args:
        dataset_name: Target dataset name
        architecture_type: Type of architecture ("MLP", "CNN", etc.)
        
    Returns:
        Tuple of (suggested_dimensions, explanation)
        
    Example:
        >>> dims, explanation = suggest_optimal_architecture('two_moons')
        >>> print(f"Suggested architecture: {dims}")
        >>> print(f"Explanation: {explanation}")
    """
    optimal_params = get_optimal_capacity(dataset_name)
    
    if architecture_type.upper() == "MLP":
        # Suggest MLP architecture that approximates optimal parameter count
        # For input_dim=2, output_dim=1, we need to find hidden dimensions
        
        if optimal_params <= 500:  # TwoMoons case
            suggested_dims = [2, 16, 16, 1]  # ~464 parameters
            explanation = "16x16 MLP optimal for TwoMoons (Phase 2D validated)"
        elif optimal_params <= 2000:  # Circles case  
            suggested_dims = [2, 32, 32, 1]  # ~1,664 parameters
            explanation = "32x32 MLP predicted optimal for medium complexity (Phase 2D extrapolated)"
        else:  # Belgium case
            suggested_dims = [2, 64, 64, 1]  # ~5,888 parameters
            explanation = "64x64 MLP predicted optimal for high complexity (Phase 2D extrapolated)"
    else:
        # For other architectures, provide general guidance
        suggested_dims = []
        explanation = f"Optimal parameter count for {dataset_name}: {optimal_params:,} parameters"
    
    return suggested_dims, explanation


# Capacity categories for analysis
CAPACITY_CATEGORIES = {
    'under_parameterized': (0.0, 0.5),
    'optimal': (0.5, 2.0), 
    'over_parameterized': (2.0, float('inf'))
}


def categorize_capacity(capacity_ratio: float) -> str:
    """
    Categorize capacity ratio based on Phase 2D findings.
    
    Args:
        capacity_ratio: Ratio of model parameters to optimal parameters
        
    Returns:
        Capacity category string
    """
    for category, (min_ratio, max_ratio) in CAPACITY_CATEGORIES.items():
        if min_ratio <= capacity_ratio < max_ratio:
            return category
    
    return 'unknown'