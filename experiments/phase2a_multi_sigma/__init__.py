"""
SPECTRA Phase 2A Multi-Sigma Experiments.

Phase 2A implements systematic characterization of σ-performance-variance 
trade-offs across datasets and applications for optimal control strategies.
"""

from .multi_sigma_experiment import Phase2AExperimentRunner, MultiSigmaResults

__all__ = [
    'Phase2AExperimentRunner',
    'MultiSigmaResults'
]