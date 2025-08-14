"""
Deterministic seed management for reproducible experiments.

Provides comprehensive random seed control across all relevant libraries
to ensure reproducible scientific results.
"""

import random
import numpy as np
import torch
from typing import Optional


def set_seed(seed: int, deterministic: bool = True) -> None:
    """
    Set random seeds for all relevant libraries for reproducible experiments.
    
    Args:
        seed: Random seed value
        deterministic: Whether to enable deterministic operations (may reduce performance)
    """
    # Python's built-in random module
    random.seed(seed)
    
    # NumPy random state
    np.random.seed(seed)
    
    # PyTorch random state
    torch.manual_seed(seed)
    
    # CUDA random state (if available)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    if deterministic:
        # Enable deterministic operations for reproducibility
        # Note: This may impact performance
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # For newer PyTorch versions
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_generator(seed: int) -> torch.Generator:
    """
    Create a PyTorch random number generator with specified seed.
    
    Args:
        seed: Random seed value
        
    Returns:
        PyTorch generator with the specified seed
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


class SeedManager:
    """
    Context manager for temporary seed setting during experiments.
    
    Allows setting seeds for specific code blocks while preserving
    the global random state afterwards.
    """
    
    def __init__(self, seed: int, deterministic: bool = True):
        """
        Initialize seed manager.
        
        Args:
            seed: Random seed to set
            deterministic: Whether to enable deterministic operations
        """
        self.seed = seed
        self.deterministic = deterministic
        
        # Store original states
        self._python_state = None
        self._numpy_state = None
        self._torch_state = None
        self._cuda_states = None
        self._original_deterministic = None
        self._original_benchmark = None
    
    def __enter__(self):
        """Enter context: save current states and set new seed."""
        # Save current random states
        self._python_state = random.getstate()
        self._numpy_state = np.random.get_state()
        self._torch_state = torch.get_rng_state()
        
        if torch.cuda.is_available():
            self._cuda_states = [torch.cuda.get_rng_state(i) 
                               for i in range(torch.cuda.device_count())]
        
        # Save deterministic settings
        self._original_deterministic = torch.backends.cudnn.deterministic
        self._original_benchmark = torch.backends.cudnn.benchmark
        
        # Set new seed
        set_seed(self.seed, self.deterministic)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context: restore original random states."""
        # Restore random states
        random.setstate(self._python_state)
        np.random.set_state(self._numpy_state)
        torch.set_rng_state(self._torch_state)
        
        if torch.cuda.is_available() and self._cuda_states is not None:
            for i, state in enumerate(self._cuda_states):
                torch.cuda.set_rng_state(state, i)
        
        # Restore deterministic settings
        torch.backends.cudnn.deterministic = self._original_deterministic
        torch.backends.cudnn.benchmark = self._original_benchmark


def create_seed_sequence(base_seed: int, count: int) -> list[int]:
    """
    Create a deterministic sequence of seeds from a base seed.
    
    Useful for generating consistent seed sequences for multi-seed experiments
    while maintaining reproducibility.
    
    Args:
        base_seed: Base seed value
        count: Number of seeds to generate
        
    Returns:
        List of deterministically generated seeds
    """
    # Use numpy's random generator for reproducible seed generation
    rng = np.random.RandomState(base_seed)
    return rng.randint(0, 2**31 - 1, size=count).tolist()


def validate_reproducibility(func, seed: int, n_runs: int = 3, **kwargs) -> bool:
    """
    Validate that a function produces reproducible results with the same seed.
    
    Args:
        func: Function to test for reproducibility
        seed: Seed to use for testing
        n_runs: Number of runs to compare
        **kwargs: Additional arguments to pass to func
        
    Returns:
        True if all runs produce identical results, False otherwise
    """
    results = []
    
    for _ in range(n_runs):
        with SeedManager(seed):
            result = func(**kwargs)
            results.append(result)
    
    # Check if all results are identical
    if isinstance(results[0], torch.Tensor):
        return all(torch.allclose(results[0], r, atol=1e-10) for r in results[1:])
    elif isinstance(results[0], np.ndarray):
        return all(np.allclose(results[0], r, atol=1e-10) for r in results[1:])
    else:
        return all(results[0] == r for r in results[1:])


def seed_worker(worker_id: int, base_seed: int) -> None:
    """
    Worker initialization function for PyTorch DataLoader.
    
    Ensures deterministic behavior in multi-process data loading.
    
    Args:
        worker_id: Worker process ID
        base_seed: Base seed for deterministic worker seeding
    """
    worker_seed = base_seed + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)