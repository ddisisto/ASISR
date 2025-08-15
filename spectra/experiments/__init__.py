"""
SPECTRA Experiments Module

Standardized experiment orchestration for all research phases.
"""

from .base import BaseExperiment, ExperimentResult, ComparisonResult
from .phase2b import Phase2BComparisonExperiment

__all__ = [
    'BaseExperiment',
    'ExperimentResult', 
    'ComparisonResult',
    'Phase2BComparisonExperiment'
]