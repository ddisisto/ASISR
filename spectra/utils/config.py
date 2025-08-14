"""
Configuration management for SPECTRA experiments.

Provides YAML-based configuration loading with validation and type checking
for reproducible scientific experiments.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
import torch


class SPECTRAConfig:
    """
    Configuration manager for SPECTRA experiments.
    
    Loads and validates YAML configuration files with standardized schemas
    for model, data, training, and experiment parameters.
    """
    
    def __init__(self, config_path: Union[str, Path]):
        """
        Initialize configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config is None:
            raise ValueError(f"Empty or invalid YAML file: {self.config_path}")
        
        return config
    
    def _validate_config(self) -> None:
        """Validate configuration structure and required fields."""
        required_sections = ['model', 'data', 'training']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model configuration
        model_config = self.config['model']
        if 'type' not in model_config:
            raise ValueError("Model configuration must specify 'type'")
        
        # Validate data configuration
        data_config = self.config['data']
        if 'type' not in data_config:
            raise ValueError("Data configuration must specify 'type'")
        
        # Validate training configuration
        training_config = self.config['training']
        required_training = ['epochs', 'learning_rate']
        for param in required_training:
            if param not in training_config:
                raise ValueError(f"Training configuration must specify '{param}'")
    
    @property
    def model(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config['model']
    
    @property
    def data(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self.config['data']
    
    @property
    def training(self) -> Dict[str, Any]:
        """Get training configuration."""
        return self.config['training']
    
    @property
    def regularization(self) -> Optional[Dict[str, Any]]:
        """Get regularization configuration (optional)."""
        return self.config.get('regularization')
    
    @property
    def experiment(self) -> Dict[str, Any]:
        """Get experiment configuration with defaults."""
        defaults = {
            'seeds': [42, 123, 456, 789, 1011],
            'metrics': ['accuracy', 'loss', 'criticality_score'],
            'save_checkpoints': False,
            'log_interval': 10
        }
        
        exp_config = self.config.get('experiment', {})
        # Merge with defaults
        for key, default_value in defaults.items():
            if key not in exp_config:
                exp_config[key] = default_value
        
        return exp_config
    
    def get_device(self) -> torch.device:
        """Get computation device based on availability and config."""
        # Check experiment_config first, then fallback to top level
        device_config = self.config.get('experiment_config', {}).get('device')
        if device_config is None:
            device_config = self.config.get('device', 'auto')
        
        if device_config == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cpu':
            return torch.device('cpu')
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            return torch.device('cuda')
        else:
            return torch.device(device_config)
    
    def get_seeds(self) -> List[int]:
        """Get list of random seeds for multi-seed experiments."""
        return self.experiment['seeds']
    
    def get_metrics(self) -> List[str]:
        """Get list of metrics to track during experiments."""
        return self.experiment['metrics']
    
    def to_dict(self) -> Dict[str, Any]:
        """Return complete configuration as dictionary."""
        return self.config.copy()
    
    def save(self, output_path: Union[str, Path]) -> None:
        """Save current configuration to YAML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"SPECTRAConfig(path={self.config_path}, model={self.model['type']}, " \
               f"data={self.data['type']}, seeds={len(self.get_seeds())})"


def load_config(config_path: Union[str, Path]) -> SPECTRAConfig:
    """
    Convenience function to load SPECTRA configuration.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Loaded and validated configuration object
    """
    return SPECTRAConfig(config_path)


def create_default_config() -> Dict[str, Any]:
    """
    Create default configuration template.
    
    Returns:
        Dictionary with default configuration structure
    """
    return {
        'model': {
            'type': 'SpectralMLP',
            'hidden_dims': [64, 64],
            'activation': 'relu'
        },
        'data': {
            'type': 'BaarleMap',
            'resolution': 200,
            'bounds': [-1.5, 2.5, -1.0, 1.5]
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-3,
            'batch_size': 1000,
            'optimizer': 'adam'
        },
        'regularization': None,
        'experiment': {
            'seeds': [42, 123, 456, 789, 1011],
            'metrics': ['accuracy', 'loss', 'criticality_score'],
            'save_checkpoints': False,
            'log_interval': 10
        },
        'device': 'auto'
    }


def save_default_config(output_path: Union[str, Path]) -> None:
    """
    Save default configuration template to YAML file.
    
    Args:
        output_path: Path where to save the default configuration
    """
    config = create_default_config()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)