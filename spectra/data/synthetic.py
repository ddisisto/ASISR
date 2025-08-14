"""
Synthetic dataset loaders for SPECTRA Phase 2A validation.

This module provides synthetic datasets with varying boundary complexities 
for systematic characterization of spectral regularization trade-offs.
"""

from typing import Tuple, Optional
import torch
import numpy as np
from sklearn.datasets import make_moons, make_circles


class SyntheticDatasetLoader:
    """Base class for synthetic dataset generation with consistent interface."""
    
    def __init__(self, n_samples: int = 1000, random_state: Optional[int] = None):
        """
        Initialize synthetic dataset loader.
        
        Args:
            n_samples: Number of samples to generate
            random_state: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_state = random_state
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate and return dataset.
        
        Returns:
            Tuple of (features, labels) as torch tensors
        """
        raise NotImplementedError("Subclasses must implement load_data()")
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """
        Get coordinate bounds for visualization.
        
        Returns:
            Tuple of (x_min, x_max, y_min, y_max)
        """
        raise NotImplementedError("Subclasses must implement get_bounds()")


class TwoMoonsLoader(SyntheticDatasetLoader):
    """
    Two interleaving half-circles dataset loader.
    
    Classic benchmark for boundary complexity with controllable noise level.
    """
    
    def __init__(self, n_samples: int = 1000, noise: float = 0.15, 
                 random_state: Optional[int] = None):
        """
        Initialize two moons dataset loader.
        
        Args:
            n_samples: Number of samples to generate
            noise: Standard deviation of Gaussian noise added to data
            random_state: Random seed for reproducibility
        """
        super().__init__(n_samples, random_state)
        self.noise = noise
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate two moons dataset."""
        X, y = make_moons(n_samples=self.n_samples, noise=self.noise, 
                         random_state=self.random_state)
        
        # Convert to torch tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        
        return X_tensor, y_tensor
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get visualization bounds for two moons dataset."""
        return (-1.5, 2.5, -1.0, 1.5)  # Standard two moons bounds


class CirclesLoader(SyntheticDatasetLoader):
    """
    Concentric circles dataset loader.
    
    Tests ability to learn non-linearly separable boundaries.
    """
    
    def __init__(self, n_samples: int = 1000, noise: float = 0.1,
                 factor: float = 0.3, random_state: Optional[int] = None):
        """
        Initialize circles dataset loader.
        
        Args:
            n_samples: Number of samples to generate
            noise: Standard deviation of Gaussian noise added to data
            factor: Scale factor between inner and outer circle (0 < factor < 1)
            random_state: Random seed for reproducibility
        """
        super().__init__(n_samples, random_state)
        self.noise = noise
        self.factor = factor
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate concentric circles dataset."""
        X, y = make_circles(n_samples=self.n_samples, noise=self.noise,
                           factor=self.factor, random_state=self.random_state)
        
        # Convert to torch tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        
        return X_tensor, y_tensor
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get visualization bounds for circles dataset."""
        return (-1.5, 1.5, -1.5, 1.5)  # Standard circles bounds


class CheckerboardLoader(SyntheticDatasetLoader):
    """
    Checkerboard pattern dataset loader.
    
    High boundary complexity test with multiple decision regions.
    """
    
    def __init__(self, n_samples: int = 1000, grid_size: int = 4,
                 noise: float = 0.1, random_state: Optional[int] = None):
        """
        Initialize checkerboard dataset loader.
        
        Args:
            n_samples: Number of samples to generate
            grid_size: Number of squares per dimension (total squares = grid_size^2)
            noise: Standard deviation of Gaussian noise added to data
            random_state: Random seed for reproducibility
        """
        super().__init__(n_samples, random_state)
        self.grid_size = grid_size
        self.noise = noise
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate checkerboard pattern dataset."""
        # Generate random points in [0, 1] x [0, 1]
        X = np.random.uniform(0, 1, size=(self.n_samples, 2))
        
        # Determine checkerboard pattern
        # Grid coordinates for each point
        grid_x = (X[:, 0] * self.grid_size).astype(int)
        grid_y = (X[:, 1] * self.grid_size).astype(int)
        
        # Checkerboard pattern: (i + j) % 2
        y = (grid_x + grid_y) % 2
        
        # Add noise
        if self.noise > 0:
            X += np.random.normal(0, self.noise, X.shape)
        
        # Scale to [-1, 1] x [-1, 1] for consistency
        X = X * 2 - 1
        
        # Convert to torch tensors
        X_tensor = torch.from_numpy(X).float()
        y_tensor = torch.from_numpy(y).long()
        
        return X_tensor, y_tensor
    
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get visualization bounds for checkerboard dataset."""
        margin = 0.2  # Add margin for noise
        return (-1 - margin, 1 + margin, -1 - margin, 1 + margin)


# Factory function for unified data loading
def create_synthetic_loader(dataset_type: str, **kwargs) -> SyntheticDatasetLoader:
    """
    Create synthetic dataset loader by type name.
    
    Args:
        dataset_type: One of 'TwoMoons', 'Circles', 'Checkerboard'
        **kwargs: Arguments passed to dataset constructor
        
    Returns:
        Initialized dataset loader
        
    Raises:
        ValueError: If dataset_type is not recognized
    """
    loaders = {
        'TwoMoons': TwoMoonsLoader,
        'Circles': CirclesLoader,
        'Checkerboard': CheckerboardLoader
    }
    
    if dataset_type not in loaders:
        available = ', '.join(loaders.keys())
        raise ValueError(f"Unknown dataset type '{dataset_type}'. Available: {available}")
    
    return loaders[dataset_type](**kwargs)