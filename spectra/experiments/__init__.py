"""
SPECTRA Experiments Module

Standardized experiment orchestration for all research phases.
"""

from .base import BaseExperiment, ExperimentResult, ComparisonResult
from .phase2b import Phase2BComparisonExperiment
from .phase4a import Phase4AExperiment, Phase4AExperimentConfig

__all__ = [
    'BaseExperiment',
    'ExperimentResult', 
    'ComparisonResult',
    'Phase2BComparisonExperiment',
    'Phase4AExperiment',
    'Phase4AExperimentConfig'
]