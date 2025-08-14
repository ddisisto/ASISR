"""
Multi-seed experiment orchestration for statistical validation.

Provides comprehensive experiment management with statistical analysis
for scientific validation of spectral regularization benefits.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats

from ..utils.config import SPECTRAConfig
from ..utils.seed import SeedManager, set_seed
from ..data import BaarleMapLoader, create_synthetic_loader
from ..models import SpectralMLP, create_boundary_mlp
from ..regularization import (
    FixedSpectralRegularizer, 
    create_edge_of_chaos_regularizer,
    create_dynamic_regularizer,
    DynamicSpectralRegularizer
)
from ..metrics import CriticalityMonitor


class ExperimentResults:
    """Container for multi-seed experiment results with statistical analysis."""
    
    def __init__(self, config: SPECTRAConfig):
        """
        Initialize results container.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.results_per_seed = {}
        self.aggregated_results = {}
        self.statistical_tests = {}
        
    def add_seed_result(self, seed: int, result: Dict[str, Any]) -> None:
        """Add results for a specific seed."""
        self.results_per_seed[seed] = result
        
    def aggregate_results(self) -> None:
        """Compute statistical aggregation across all seeds."""
        if not self.results_per_seed:
            return
            
        # Extract metrics across seeds (interval-based)
        all_metrics = {}
        for seed, result in self.results_per_seed.items():
            for metric_name, values in result['metrics_history'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(values)
        
        # Extract trajectory metrics across seeds (every epoch)
        all_trajectory_metrics = {}
        for seed, result in self.results_per_seed.items():
            if 'trajectory_metrics' in result:
                for metric_name, values in result['trajectory_metrics'].items():
                    if metric_name not in all_trajectory_metrics:
                        all_trajectory_metrics[metric_name] = []
                    all_trajectory_metrics[metric_name].append(values)
        
        # Compute statistics for each metric (interval-based)
        self.aggregated_results = {}
        for metric_name, seed_values in all_metrics.items():
            if not seed_values:
                continue
            # Convert to numpy array for easier computation
            values_array = np.array(seed_values)  # Shape: (n_seeds, n_logged_epochs)
            
            self.aggregated_results[metric_name] = {
                'mean': np.mean(values_array, axis=0),
                'std': np.std(values_array, axis=0, ddof=1),
                'min': np.min(values_array, axis=0),
                'max': np.max(values_array, axis=0),
                'final_mean': np.mean(values_array[:, -1]),
                'final_std': np.std(values_array[:, -1], ddof=1),
                'final_values': values_array[:, -1]
            }
        
        # Compute statistics for trajectory metrics (every epoch - for visualization)
        for metric_name, seed_values in all_trajectory_metrics.items():
            if not seed_values:
                continue
            # Convert to numpy array for easier computation
            values_array = np.array(seed_values)  # Shape: (n_seeds, n_epochs)
            
            # Add trajectory-specific statistics with _trajectory suffix
            trajectory_key = metric_name
            self.aggregated_results[trajectory_key] = {
                'trajectory_mean': np.mean(values_array, axis=0),
                'trajectory_std': np.std(values_array, axis=0, ddof=1),
                'trajectory_min': np.min(values_array, axis=0),
                'trajectory_max': np.max(values_array, axis=0),
                'final_mean': np.mean(values_array[:, -1]),
                'final_std': np.std(values_array[:, -1], ddof=1),
                'final_values': values_array[:, -1]
            }
    
    def compute_confidence_intervals(self, confidence: float = 0.95) -> None:
        """Compute confidence intervals for final metrics."""
        if not self.aggregated_results:
            self.aggregate_results()
            
        alpha = 1 - confidence
        
        for metric_name, stats_dict in self.aggregated_results.items():
            final_values = stats_dict['final_values']
            n = len(final_values)
            
            if n > 1:
                # Use t-distribution for small samples
                t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
                margin_error = t_critical * stats_dict['final_std'] / np.sqrt(n)
                
                stats_dict['confidence_interval'] = {
                    'confidence': confidence,
                    'lower': stats_dict['final_mean'] - margin_error,
                    'upper': stats_dict['final_mean'] + margin_error,
                    'margin_error': margin_error
                }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics across all seeds."""
        if not self.aggregated_results:
            self.aggregate_results()
            
        summary = {
            'n_seeds': len(self.results_per_seed),
            'config': self.config.to_dict(),
            'final_metrics': {}
        }
        
        for metric_name, stats_dict in self.aggregated_results.items():
            summary['final_metrics'][metric_name] = {
                'mean': float(stats_dict['final_mean']),
                'std': float(stats_dict['final_std']),
                'confidence_interval': stats_dict.get('confidence_interval', {})
            }
            
        return summary


class SPECTRAExperiment:
    """
    Multi-seed experiment orchestrator for SPECTRA research.
    
    Manages complete experimental workflows including model training,
    evaluation, and statistical analysis across multiple random seeds.
    """
    
    def __init__(self, config: SPECTRAConfig):
        """
        Initialize experiment with configuration.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.device = config.get_device()
        self.results = ExperimentResults(config)
        
        # Initialize shared components
        self.data_loader = None
        self.criticality_monitor = CriticalityMonitor()
        
    def _setup_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Setup data loading based on configuration."""
        data_config = self.config.data
        data_type = data_config['type']
        
        if data_type == 'BaarleMap':
            if self.data_loader is None:
                self.data_loader = BaarleMapLoader()
            
            resolution = data_config.get('resolution', 200)
            coords, labels = self.data_loader.get_torch_tensors(resolution=resolution)
            
            return coords.to(self.device), labels.to(self.device)
            
        elif data_type in ['TwoMoons', 'Circles', 'Checkerboard']:
            # Synthetic dataset loading
            loader_kwargs = {k: v for k, v in data_config.items() if k != 'type'}
            synthetic_loader = create_synthetic_loader(data_type, **loader_kwargs)
            coords, labels = synthetic_loader.load_data()
            
            return coords.to(self.device), labels.to(self.device)
            
        else:
            raise ValueError(f"Unsupported data type: {data_type}. "
                           f"Supported: BaarleMap, TwoMoons, Circles, Checkerboard")
    
    def _create_model(self) -> SpectralMLP:
        """Create model based on configuration."""
        model_config = self.config.model
        
        if model_config['type'] == 'SpectralMLP':
            hidden_dims = model_config.get('hidden_dims', [64, 64])
            model = create_boundary_mlp(hidden_dims=hidden_dims)
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
            
        return model.to(self.device)
    
    def _create_regularizer(self) -> Optional[Union[FixedSpectralRegularizer, DynamicSpectralRegularizer]]:
        """Create regularizer based on configuration."""
        reg_config = self.config.regularization
        
        if reg_config is None:
            return None
        
        if reg_config['type'] == 'fixed_spectral':
            strength = reg_config.get('strength', 0.1)
            target_sigma = reg_config.get('target_sigma', 1.0)
            return FixedSpectralRegularizer(
                target_sigma=target_sigma,
                regularization_strength=strength,
                power_iterations=10
            )
        elif reg_config['type'] in ['linear_schedule', 'exponential_schedule', 'step_schedule', 'adaptive_schedule']:
            # Dynamic scheduling parameters
            initial_sigma = reg_config.get('initial_sigma', 2.5)
            final_sigma = reg_config.get('final_sigma', 1.0)
            total_epochs = self.config.training['epochs']
            strength = reg_config.get('strength', 0.1)
            
            # Schedule-specific parameters
            schedule_kwargs = {
                'initial_sigma': initial_sigma,
                'final_sigma': final_sigma,
                'total_epochs': total_epochs,
                'regularization_strength': strength
            }
            
            if reg_config['type'] == 'exponential_schedule':
                schedule_kwargs['decay_rate'] = reg_config.get('decay_rate', 5.0)
            elif reg_config['type'] == 'step_schedule':
                sigma_schedule = reg_config.get('sigma_schedule', [(0, initial_sigma), (total_epochs//2, final_sigma)])
                schedule_kwargs = {
                    'sigma_schedule': sigma_schedule,
                    'total_epochs': total_epochs,
                    'regularization_strength': strength
                }
            elif reg_config['type'] == 'adaptive_schedule':
                schedule_kwargs['adaptation_rate'] = reg_config.get('adaptation_rate', 0.1)
                schedule_kwargs['warmup_epochs'] = reg_config.get('warmup_epochs', 10)
            
            # Extract schedule type
            schedule_type = reg_config['type'].replace('_schedule', '')
            return create_dynamic_regularizer(schedule_type, **schedule_kwargs)
        else:
            raise ValueError(f"Unsupported regularization type: {reg_config['type']}")
    
    def _create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer based on configuration."""
        training_config = self.config.training
        lr = training_config['learning_rate']
        optimizer_type = training_config.get('optimizer', 'adam')
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_type.lower() == 'sgd':
            momentum = training_config.get('momentum', 0.9)
            return optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    def run_single_seed(self, seed: int) -> Dict[str, Any]:
        """
        Run experiment for a single seed.
        
        Args:
            seed: Random seed for this run
            
        Returns:
            Dictionary containing training metrics and final results
        """
        with SeedManager(seed):
            # Setup experiment components
            coords, labels = self._setup_data()
            model = self._create_model()
            regularizer = self._create_regularizer()
            optimizer = self._create_optimizer(model)
            criterion = nn.BCEWithLogitsLoss()
            
            # Training configuration
            training_config = self.config.training
            epochs = training_config['epochs']
            batch_size = training_config.get('batch_size', len(coords))
            log_interval = self.config.experiment.get('log_interval', 10)
            
            # Metrics tracking - separate detailed vs summary logging
            metrics_history = {
                'epoch': [],
                'train_loss': [],
                'spectral_loss': [],
                'total_loss': [],
                'accuracy': [],
                'dead_neuron_rate': [],
                'perturbation_sensitivity': [],
                'spectral_radius_avg': [],
                'boundary_fractal_dim': [],
                'criticality_score': []
            }
            
            # Full trajectory tracking (every epoch for visualization)
            trajectory_metrics = {
                'epoch': [],
                'train_loss': [],
                'spectral_loss': [], 
                'total_loss': [],
                'accuracy': [],
                'criticality_score': [],
                'spectral_radius_avg': [],
                'boundary_fractal_dim': [],
                'sigma_target': []  # Track σ evolution for dynamic regularizers
            }
            
            # Convert labels for BCE loss
            labels_float = labels.float().unsqueeze(1)
            
            # Training loop
            start_time = time.time()
            
            for epoch in range(epochs):
                # Progress reporting every 25 epochs (with accuracy)
                progress_interval = 25
                if epoch % progress_interval == 0 or epoch == epochs - 1:
                    # Quick accuracy check for progress reporting
                    model.eval()
                    with torch.no_grad():
                        test_output = model(coords)
                        test_predictions = (torch.sigmoid(test_output) >= 0.5).long().squeeze(1)
                        current_accuracy = (test_predictions == labels).float().mean().item()
                    print(f"  Seed {seed} - Epoch {epoch}/{epochs} - Accuracy: {current_accuracy:.3f}")
                    model.train()
                
                # Update dynamic regularizer if needed
                if regularizer is not None and isinstance(regularizer, DynamicSpectralRegularizer):
                    regularizer.update_epoch(epoch)
                
                model.train()
                
                # Forward pass
                output = model(coords)
                task_loss = criterion(output, labels_float)
                
                # Spectral regularization
                if regularizer is not None:
                    spectral_loss = model.spectral_loss(regularizer)
                else:
                    spectral_loss = torch.tensor(0.0, device=self.device)
                
                total_loss = task_loss + spectral_loss
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                
                # Update adaptive regularizer with training metrics if needed
                if (regularizer is not None and 
                    hasattr(regularizer, 'update_training_metrics')):
                    regularizer.update_training_metrics(total_loss.item())
                
                # Trajectory logging (every epoch for visualization)
                with torch.no_grad():
                    # Fast accuracy computation for trajectory
                    predictions = (torch.sigmoid(output) >= 0.5).long().squeeze(1)
                    accuracy = (predictions == labels).float().mean().item()
                    
                    # Track σ target for dynamic regularizers
                    sigma_target = None
                    if regularizer is not None and hasattr(regularizer, 'current_sigma'):
                        sigma_target = regularizer.current_sigma
                    elif regularizer is not None and hasattr(regularizer, 'get_targets'):
                        targets = regularizer.get_targets()
                        sigma_target = list(targets.values())[0] if targets else None
                    
                    # Compute lightweight criticality metrics for trajectory
                    model.eval()
                    with torch.no_grad():
                        predictions = (torch.sigmoid(output) >= 0.5).long().squeeze(1)
                        accuracy = (predictions == labels).float().mean().item()
                        
                        # Quick criticality assessment (smaller sample for efficiency)
                        sample_size = min(200, len(coords))
                        sample_coords = coords[:sample_size]
                        criticality_metrics = self.criticality_monitor.assess_criticality(
                            model, sample_coords
                        )
                        criticality_score = self.criticality_monitor.criticality_score(
                            criticality_metrics
                        )
                    
                    # Store trajectory data (every epoch)
                    trajectory_metrics['epoch'].append(epoch)
                    trajectory_metrics['train_loss'].append(task_loss.item())
                    trajectory_metrics['spectral_loss'].append(spectral_loss.item())
                    trajectory_metrics['total_loss'].append(total_loss.item())
                    trajectory_metrics['accuracy'].append(accuracy)
                    trajectory_metrics['criticality_score'].append(criticality_score)
                    trajectory_metrics['spectral_radius_avg'].append(criticality_metrics['spectral_radius_avg'])
                    trajectory_metrics['boundary_fractal_dim'].append(criticality_metrics['boundary_fractal_dim'])
                    trajectory_metrics['sigma_target'].append(sigma_target)
                
                # Evaluation metrics
                if epoch % log_interval == 0 or epoch == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        # Accuracy
                        predictions = (torch.sigmoid(output) >= 0.5).long().squeeze(1)
                        accuracy = (predictions == labels).float().mean().item()
                        
                        # Criticality assessment (use subset for efficiency)
                        sample_size = min(500, len(coords))
                        sample_coords = coords[:sample_size]
                        criticality_metrics = self.criticality_monitor.assess_criticality(
                            model, sample_coords
                        )
                        criticality_score = self.criticality_monitor.criticality_score(
                            criticality_metrics
                        )
                        
                        # Store metrics
                        metrics_history['epoch'].append(epoch)
                        metrics_history['train_loss'].append(task_loss.item())
                        metrics_history['spectral_loss'].append(spectral_loss.item())
                        metrics_history['total_loss'].append(total_loss.item())
                        metrics_history['accuracy'].append(accuracy)
                        metrics_history['dead_neuron_rate'].append(
                            criticality_metrics['dead_neuron_rate']
                        )
                        metrics_history['perturbation_sensitivity'].append(
                            criticality_metrics['perturbation_sensitivity']
                        )
                        metrics_history['spectral_radius_avg'].append(
                            criticality_metrics['spectral_radius_avg']
                        )
                        metrics_history['boundary_fractal_dim'].append(
                            criticality_metrics['boundary_fractal_dim']
                        )
                        metrics_history['criticality_score'].append(criticality_score)
            
            training_time = time.time() - start_time
            
            return {
                'seed': seed,
                'metrics_history': metrics_history,
                'trajectory_metrics': trajectory_metrics,  # Added for visualization
                'final_accuracy': accuracy,
                'final_criticality_score': criticality_score,
                'training_time': training_time,
                'model_state': model.state_dict(),
                'config': self.config.to_dict()
            }
    
    def run_multi_seed(self, seeds: Optional[List[int]] = None) -> ExperimentResults:
        """
        Run experiment across multiple seeds with statistical analysis.
        
        Args:
            seeds: List of seeds to use (defaults to config seeds)
            
        Returns:
            Complete experiment results with statistical analysis
        """
        if seeds is None:
            seeds = self.config.get_seeds()
        
        print(f"Running multi-seed experiment with {len(seeds)} seeds: {seeds}")
        print(f"Model: {self.config.model['type']}, Data: {self.config.data['type']}")
        print(f"Regularization: {self.config.regularization}")
        print(f"Device: {self.device}")
        print("=" * 50)
        
        for i, seed in enumerate(seeds):
            print(f"Seed {i+1}/{len(seeds)}: {seed}")
            start_time = time.time()
            
            try:
                result = self.run_single_seed(seed)
                self.results.add_seed_result(seed, result)
                
                elapsed = time.time() - start_time
                final_acc = result['final_accuracy']
                final_crit = result['final_criticality_score']
                
                print(f"  Completed in {elapsed:.1f}s - "
                      f"Accuracy: {final_acc:.3f}, "
                      f"Criticality: {final_crit:.3f}")
                
            except Exception as e:
                print(f"  FAILED: {e}")
                raise
        
        # Compute statistical analysis
        self.results.aggregate_results()
        self.results.compute_confidence_intervals()
        
        print("\n" + "=" * 50)
        print("Multi-seed experiment completed!")
        
        # Print summary
        summary = self.results.get_summary()
        for metric, stats in summary['final_metrics'].items():
            mean = stats['mean']
            std = stats['std']
            ci = stats.get('confidence_interval', {})
            
            if ci:
                print(f"{metric}: {mean:.4f} ± {std:.4f} "
                      f"(95% CI: [{ci['lower']:.4f}, {ci['upper']:.4f}])")
            else:
                print(f"{metric}: {mean:.4f} ± {std:.4f}")
        
        return self.results


def compare_experiments(baseline_results: ExperimentResults, 
                       spectral_results: ExperimentResults,
                       metric: str = 'accuracy') -> Dict[str, Any]:
    """
    Statistical comparison between baseline and spectral regularization experiments.
    
    Args:
        baseline_results: Results from baseline experiment
        spectral_results: Results from spectral regularization experiment
        metric: Metric to compare (default: 'accuracy')
        
    Returns:
        Statistical comparison results including effect size and significance tests
    """
    if metric not in baseline_results.aggregated_results:
        raise ValueError(f"Metric '{metric}' not found in baseline results")
    if metric not in spectral_results.aggregated_results:
        raise ValueError(f"Metric '{metric}' not found in spectral results")
    
    baseline_values = baseline_results.aggregated_results[metric]['final_values']
    spectral_values = spectral_results.aggregated_results[metric]['final_values']
    
    # Statistical tests
    t_stat, p_value = stats.ttest_ind(baseline_values, spectral_values)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(baseline_values) - 1) * np.var(baseline_values, ddof=1) +
                         (len(spectral_values) - 1) * np.var(spectral_values, ddof=1)) /
                        (len(baseline_values) + len(spectral_values) - 2))
    
    cohens_d = (np.mean(spectral_values) - np.mean(baseline_values)) / pooled_std
    
    return {
        'metric': metric,
        'baseline_mean': np.mean(baseline_values),
        'baseline_std': np.std(baseline_values, ddof=1),
        'spectral_mean': np.mean(spectral_values),
        'spectral_std': np.std(spectral_values, ddof=1),
        'difference': np.mean(spectral_values) - np.mean(baseline_values),
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'effect_size_interpretation': _interpret_effect_size(abs(cohens_d)),
        'significant': p_value < 0.05
    }


def _interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    if d < 0.2:
        return "negligible"
    elif d < 0.5:
        return "small"
    elif d < 0.8:
        return "medium"
    else:
        return "large"