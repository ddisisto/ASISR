"""
Real dataset loaders for SPECTRA Phase 4A realistic scale testing.

Provides standard benchmark datasets like MNIST, FashionMNIST, etc.
with consistent interface for integration with SPECTRA framework.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import numpy as np
from pathlib import Path
import os

try:
    import torchvision
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False


class RealDatasetLoader:
    """Base class for real dataset loading with consistent interface."""
    
    def __init__(self, 
                 normalize: bool = True,
                 flatten: bool = False,
                 data_dir: Optional[str] = None):
        """
        Initialize real dataset loader.
        
        Args:
            normalize: Whether to normalize pixel values to [0,1]
            flatten: Whether to flatten images for MLP compatibility
            data_dir: Directory to store/load datasets
        """
        self.normalize = normalize
        self.flatten = flatten
        self.data_dir = data_dir or os.path.expanduser("~/datasets")
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return dataset.
        
        Returns:
            Tuple of (features, labels) as torch tensors
        """
        raise NotImplementedError("Subclasses must implement load_data()")
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get dataset information.
        
        Returns:
            Dictionary with dataset metadata
        """
        raise NotImplementedError("Subclasses must implement get_dataset_info()")


class MNISTLoader(RealDatasetLoader):
    """
    MNIST handwritten digits dataset loader.
    
    28x28 grayscale images of digits 0-9, 60k training samples.
    Standard benchmark for basic image classification.
    """
    
    def __init__(self, 
                 train: bool = True,
                 normalize: bool = True,
                 flatten: bool = True,  # Default to flatten for MLP
                 data_dir: Optional[str] = None):
        """
        Initialize MNIST loader.
        
        Args:
            train: Whether to load training set (vs test set)
            normalize: Whether to normalize pixel values
            flatten: Whether to flatten images (required for MLP)
            data_dir: Directory to store datasets
        """
        super().__init__(normalize, flatten, data_dir)
        self.train = train
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for MNIST loading. Install with: pip install torchvision")
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load MNIST dataset."""
        
        # Set up transforms
        transform_list = []
        if self.normalize:
            transform_list.append(transforms.ToTensor())  # Converts to [0,1] and CHW format
        else:
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Lambda(lambda x: x * 255))  # Keep in [0,255]
        
        if self.flatten:
            transform_list.append(transforms.Lambda(lambda x: x.view(-1)))  # Flatten to 784
        
        transform = transforms.Compose(transform_list)
        
        # Load dataset
        dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=self.train,
            download=True,
            transform=transform
        )
        
        # Convert to tensors
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        features, labels = next(iter(data_loader))
        
        return features, labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get MNIST dataset information."""
        return {
            "name": "MNIST",
            "description": "Handwritten digits 0-9",
            "image_size": (28, 28),
            "channels": 1,
            "num_classes": 10,
            "num_samples": 60000 if self.train else 10000,
            "input_dim": 784 if self.flatten else (1, 28, 28),
            "complexity": "low"  # Relatively simple benchmark
        }


class FashionMNISTLoader(RealDatasetLoader):
    """
    Fashion-MNIST clothing items dataset loader.
    
    28x28 grayscale images of clothing items, 60k training samples.
    Drop-in replacement for MNIST with higher complexity.
    """
    
    def __init__(self, 
                 train: bool = True,
                 normalize: bool = True,
                 flatten: bool = True,
                 data_dir: Optional[str] = None):
        super().__init__(normalize, flatten, data_dir)
        self.train = train
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for Fashion-MNIST loading")
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load Fashion-MNIST dataset."""
        
        transform_list = []
        if self.normalize:
            transform_list.append(transforms.ToTensor())
        else:
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Lambda(lambda x: x * 255))
        
        if self.flatten:
            transform_list.append(transforms.Lambda(lambda x: x.view(-1)))
        
        transform = transforms.Compose(transform_list)
        
        dataset = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=self.train,
            download=True,
            transform=transform
        )
        
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        features, labels = next(iter(data_loader))
        
        return features, labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get Fashion-MNIST dataset information."""
        return {
            "name": "Fashion-MNIST",
            "description": "Clothing items (T-shirt, trouser, pullover, etc.)",
            "image_size": (28, 28),
            "channels": 1,
            "num_classes": 10,
            "num_samples": 60000 if self.train else 10000,
            "input_dim": 784 if self.flatten else (1, 28, 28),
            "complexity": "medium"
        }


class CIFAR10Loader(RealDatasetLoader):
    """
    CIFAR-10 natural images dataset loader.
    
    32x32 color images of objects, 50k training samples.
    Higher complexity benchmark for realistic testing.
    """
    
    def __init__(self, 
                 train: bool = True,
                 normalize: bool = True,
                 flatten: bool = True,
                 data_dir: Optional[str] = None):
        super().__init__(normalize, flatten, data_dir)
        self.train = train
        
        if not TORCHVISION_AVAILABLE:
            raise ImportError("torchvision required for CIFAR-10 loading")
    
    def load_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load CIFAR-10 dataset."""
        
        transform_list = []
        if self.normalize:
            transform_list.append(transforms.ToTensor())  # Converts to [0,1]
        else:
            transform_list.append(transforms.ToTensor())
            transform_list.append(transforms.Lambda(lambda x: x * 255))
        
        if self.flatten:
            transform_list.append(transforms.Lambda(lambda x: x.view(-1)))  # Flatten to 3072
        
        transform = transforms.Compose(transform_list)
        
        dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir,
            train=self.train,
            download=True,
            transform=transform
        )
        
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset), shuffle=False)
        features, labels = next(iter(data_loader))
        
        return features, labels
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get CIFAR-10 dataset information."""
        return {
            "name": "CIFAR-10", 
            "description": "Natural images (airplane, automobile, bird, etc.)",
            "image_size": (32, 32),
            "channels": 3,
            "num_classes": 10,
            "num_samples": 50000 if self.train else 10000,
            "input_dim": 3072 if self.flatten else (3, 32, 32),
            "complexity": "high"
        }


def create_real_dataset_loader(dataset_name: str, **kwargs) -> RealDatasetLoader:
    """
    Factory function to create real dataset loaders.
    
    Args:
        dataset_name: Name of dataset ('MNIST', 'FashionMNIST', 'CIFAR10')
        **kwargs: Additional arguments passed to loader
        
    Returns:
        Configured dataset loader
        
    Raises:
        ValueError: If dataset_name not recognized
    """
    dataset_map = {
        'MNIST': MNISTLoader,
        'FashionMNIST': FashionMNISTLoader,
        'CIFAR10': CIFAR10Loader
    }
    
    if dataset_name not in dataset_map:
        available = ', '.join(dataset_map.keys())
        raise ValueError(f"Unknown dataset '{dataset_name}'. Available: {available}")
    
    return dataset_map[dataset_name](**kwargs)


# Compatibility aliases for common use
MnistLoader = MNISTLoader  # Common alternative spelling
Cifar10Loader = CIFAR10Loader  # Common alternative spelling