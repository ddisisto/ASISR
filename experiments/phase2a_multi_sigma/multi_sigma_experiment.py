"""
Phase 2A Multi-Sigma Experiment Framework.

Extends the unified experiment system to support systematic characterization 
of σ-performance-variance relationships across datasets and applications.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from spectra.utils.config import load_config, SPECTRAConfig
from spectra.training.experiment import SPECTRAExperiment, ExperimentResults
from spectra.data import create_synthetic_loader
import torch


class MultiSigmaResults:
    """Container for multi-σ sweep experiment results."""
    
    def __init__(self, config: SPECTRAConfig, sigma_values: List[float]):
        """
        Initialize multi-sigma results container.
        
        Args:
            config: Base experiment configuration
            sigma_values: List of σ values tested
        """
        self.config = config
        self.sigma_values = sigma_values
        self.results_per_sigma = {}  # σ -> ExperimentResults
        self.trade_off_curves = {}
        
    def add_sigma_result(self, sigma: float, results: ExperimentResults) -> None:
        """Add results for a specific sigma value."""
        self.results_per_sigma[sigma] = results
        
    def compute_trade_off_curves(self) -> None:
        """Compute performance-variance trade-off curves across σ values."""
        metrics = ['accuracy', 'criticality_score', 'boundary_fractal_dim', 'spectral_radius_avg']
        
        self.trade_off_curves = {}
        for metric in metrics:
            self.trade_off_curves[metric] = {
                'sigma_values': [],
                'means': [],
                'stds': [],
                'confidence_lower': [],
                'confidence_upper': []
            }
        
        # Extract data for each sigma
        for sigma in sorted(self.sigma_values):
            if sigma not in self.results_per_sigma:
                continue
                
            results = self.results_per_sigma[sigma]
            if not results.aggregated_results:
                results.aggregate_results()
                results.compute_confidence_intervals()
            
            for metric in metrics:
                if metric in results.aggregated_results:
                    agg = results.aggregated_results[metric]
                    self.trade_off_curves[metric]['sigma_values'].append(sigma)
                    self.trade_off_curves[metric]['means'].append(agg['final_mean'])
                    self.trade_off_curves[metric]['stds'].append(agg['final_std'])
                    
                    if 'confidence_interval' in agg:
                        ci = agg['confidence_interval']
                        self.trade_off_curves[metric]['confidence_lower'].append(ci['lower'])
                        self.trade_off_curves[metric]['confidence_upper'].append(ci['upper'])
                    else:
                        # Fallback to mean ± std
                        self.trade_off_curves[metric]['confidence_lower'].append(
                            agg['final_mean'] - agg['final_std']
                        )
                        self.trade_off_curves[metric]['confidence_upper'].append(
                            agg['final_mean'] + agg['final_std']
                        )
    
    def find_optimal_sigma(self, metric: str = 'accuracy', 
                          constraint: str = None) -> Tuple[float, float]:
        """
        Find optimal σ value based on specified criteria.
        
        Args:
            metric: Primary metric to optimize ('accuracy', 'criticality_score', etc.)
            constraint: Additional constraint ('low_variance', 'high_performance')
            
        Returns:
            Tuple of (optimal_sigma, optimal_value)
        """
        if metric not in self.trade_off_curves:
            raise ValueError(f"Metric '{metric}' not available in trade-off curves")
        
        curve = self.trade_off_curves[metric]
        sigmas = np.array(curve['sigma_values'])
        means = np.array(curve['means'])
        stds = np.array(curve['stds'])
        
        if constraint == 'low_variance':
            # Find sigma with best performance among low-variance options
            # Define low variance as bottom 25th percentile
            var_threshold = np.percentile(stds, 25)
            low_var_mask = stds <= var_threshold
            if np.any(low_var_mask):
                low_var_idx = np.argmax(means[low_var_mask])
                optimal_idx = np.where(low_var_mask)[0][low_var_idx]
            else:
                optimal_idx = np.argmax(means)
                
        elif constraint == 'high_performance':
            # Find sigma with lowest variance among high-performance options
            perf_threshold = np.percentile(means, 75)
            high_perf_mask = means >= perf_threshold
            if np.any(high_perf_mask):
                high_perf_idx = np.argmin(stds[high_perf_mask])
                optimal_idx = np.where(high_perf_mask)[0][high_perf_idx]
            else:
                optimal_idx = np.argmax(means)
                
        else:
            # Default: maximize performance
            optimal_idx = np.argmax(means)
        
        return sigmas[optimal_idx], means[optimal_idx]


class Phase2AExperimentRunner:
    """
    Phase 2A experiment orchestrator for multi-σ trade-off characterization.
    
    Systematically evaluates spectral regularization trade-offs across
    different σ values and datasets to build application-specific frameworks.
    """
    
    def __init__(self):
        """Initialize Phase 2A experiment runner."""
        pass
        
    def run_multi_sigma_sweep(self, config_path: str, 
                             experiment_name: str = None) -> MultiSigmaResults:
        """
        Run complete multi-σ sweep experiment.
        
        Args:
            config_path: Path to experiment configuration with multi_sigma section
            experiment_name: Optional name for this experiment
            
        Returns:
            Complete multi-sigma results with trade-off analysis
        """
        # Load configuration
        config = load_config(config_path)
        
        # Extract multi-sigma configuration
        multi_sigma_config = config.config.get('multi_sigma', {})
        if not multi_sigma_config.get('enabled', False):
            raise ValueError("Multi-sigma sweep not enabled in configuration")
        
        sigma_values = multi_sigma_config['sigma_values']
        include_baseline = multi_sigma_config.get('include_baseline', True)
        
        if include_baseline:
            sigma_values = [float('inf')] + list(sigma_values)  # inf = no regularization
        
        print(f"\n{'='*80}")
        print(f"SPECTRA Phase 2A Multi-σ Trade-off Characterization")
        print(f"Dataset: {config.data['type']}")
        print(f"σ Values: {[f'{s:.1f}' if s != float('inf') else 'baseline' for s in sigma_values]}")
        print(f"Seeds: {len(config.get_seeds())} per σ value")
        print(f"{'='*80}")
        
        # Initialize results container
        results = MultiSigmaResults(config, sigma_values)
        
        # Run experiments for each sigma value
        for i, sigma in enumerate(sigma_values):
            print(f"\n{'='*60}")
            if sigma == float('inf'):
                print(f"Running Baseline Experiment (no regularization)")
                sigma_name = "baseline"
                # Disable regularization for baseline
                config.config['regularization'] = None
            else:
                print(f"Running σ = {sigma:.1f} Experiment ({i}/{len(sigma_values)-1 if include_baseline else len(sigma_values)})")
                sigma_name = f"sigma_{sigma:.1f}"
                # Set sigma value in regularization config
                if config.config['regularization'] is None:
                    config.config['regularization'] = {
                        'type': 'fixed_spectral',
                        'strength': 0.1,
                        'target_sigma': sigma
                    }
                else:
                    config.config['regularization']['target_sigma'] = sigma
            
            print(f"Configuration: {config_path}")
            print(f"{'='*60}")
            
            # Run experiment
            experiment = SPECTRAExperiment(config)
            sigma_results = experiment.run_multi_seed(seeds=config.get_seeds())
            
            # Store results
            results.add_sigma_result(sigma, sigma_results)
            
            # Print brief summary
            sigma_results.aggregate_results()
            sigma_results.compute_confidence_intervals()
            acc_stats = sigma_results.aggregated_results.get('accuracy', {})
            if acc_stats:
                print(f"Final Accuracy: {acc_stats['final_mean']:.3f} ± {acc_stats['final_std']:.3f}")
        
        # Compute trade-off curves
        print(f"\n{'='*40}")
        print("TRADE-OFF ANALYSIS")
        print(f"{'='*40}")
        results.compute_trade_off_curves()
        
        # Find optimal sigma values for different criteria
        try:
            opt_perf_sigma, opt_perf_val = results.find_optimal_sigma('accuracy')
            print(f"Optimal σ for performance: {opt_perf_sigma:.1f} (accuracy = {opt_perf_val:.3f})")
            
            opt_stable_sigma, opt_stable_val = results.find_optimal_sigma('accuracy', 'low_variance')
            print(f"Optimal σ for stability: {opt_stable_sigma:.1f} (accuracy = {opt_stable_val:.3f})")
            
        except Exception as e:
            print(f"Trade-off optimization failed: {e}")
        
        return results
    
    def run_dataset_comparison(self, configs: List[str],
                              dataset_names: Optional[List[str]] = None) -> Dict[str, MultiSigmaResults]:
        """
        Run multi-σ sweeps across multiple datasets for comparison.
        
        Args:
            configs: List of configuration file paths
            dataset_names: Optional custom names for datasets
            
        Returns:
            Dictionary mapping dataset names to multi-sigma results
        """
        if dataset_names is None:
            dataset_names = [f"dataset_{i}" for i in range(len(configs))]
        
        if len(configs) != len(dataset_names):
            raise ValueError("Number of configs must match number of dataset names")
        
        print(f"\n{'='*80}")
        print("SPECTRA Phase 2A Dataset Comparison")
        print(f"Datasets: {', '.join(dataset_names)}")
        print(f"{'='*80}")
        
        all_results = {}
        
        for config_path, dataset_name in zip(configs, dataset_names):
            print(f"\n{'='*60}")
            print(f"Processing Dataset: {dataset_name}")
            print(f"{'='*60}")
            
            results = self.run_multi_sigma_sweep(config_path, dataset_name)
            all_results[dataset_name] = results
        
        # Cross-dataset analysis
        print(f"\n{'='*40}")
        print("CROSS-DATASET ANALYSIS")
        print(f"{'='*40}")
        
        for dataset_name, results in all_results.items():
            try:
                opt_sigma, opt_val = results.find_optimal_sigma('accuracy')
                print(f"{dataset_name}: Optimal σ = {opt_sigma:.1f} (accuracy = {opt_val:.3f})")
            except Exception as e:
                print(f"{dataset_name}: Analysis failed - {e}")
        
        return all_results
    
    def generate_trade_off_plots(self, results: MultiSigmaResults,
                               output_dir: str = "plots/phase2a") -> None:
        """
        Generate publication-quality trade-off characterization plots.
        
        Args:
            results: Multi-sigma experiment results
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if not results.trade_off_curves:
            results.compute_trade_off_curves()
        
        # Set style for publication quality
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Create subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('SPECTRA Phase 2A: Multi-σ Trade-off Characterization', fontsize=16)
        
        metrics_info = {
            'accuracy': {'title': 'Performance vs σ', 'ylabel': 'Accuracy', 'ax': axes[0, 0]},
            'criticality_score': {'title': 'Criticality vs σ', 'ylabel': 'Criticality Score', 'ax': axes[0, 1]},
            'boundary_fractal_dim': {'title': 'Boundary Complexity vs σ', 'ylabel': 'Fractal Dimension', 'ax': axes[1, 0]},
            'spectral_radius_avg': {'title': 'Spectral Control vs σ', 'ylabel': 'Average Spectral Radius', 'ax': axes[1, 1]}
        }
        
        for metric, info in metrics_info.items():
            if metric not in results.trade_off_curves:
                continue
                
            curve = results.trade_off_curves[metric]
            ax = info['ax']
            
            # Plot mean with confidence intervals
            sigma_values = np.array(curve['sigma_values'])
            means = np.array(curve['means'])
            lower = np.array(curve['confidence_lower'])
            upper = np.array(curve['confidence_upper'])
            
            # Handle baseline (σ = inf) for plotting
            plot_sigmas = ['baseline' if s == float('inf') else f'{s:.1f}' 
                          for s in sigma_values]
            
            ax.plot(range(len(means)), means, 'o-', linewidth=2, markersize=6)
            ax.fill_between(range(len(means)), lower, upper, alpha=0.3)
            
            ax.set_title(info['title'], fontsize=12, fontweight='bold')
            ax.set_ylabel(info['ylabel'], fontsize=11)
            ax.set_xlabel('σ Value', fontsize=11)
            ax.set_xticks(range(len(plot_sigmas)))
            ax.set_xticklabels(plot_sigmas, rotation=45)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'multi_sigma_trade_offs.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Trade-off plots saved to {output_path}/multi_sigma_trade_offs.png")


def main():
    """Main entry point for Phase 2A multi-sigma experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPECTRA Phase 2A Multi-σ Experiment Runner")
    parser.add_argument('mode', choices=['single', 'comparison'],
                       help='Experiment mode: single dataset or cross-dataset comparison')
    parser.add_argument('--config', required=True,
                       help='Configuration file for single mode')
    parser.add_argument('--configs', nargs='+',
                       help='Configuration files for comparison mode')
    parser.add_argument('--names', nargs='+',
                       help='Dataset names for comparison mode')
    parser.add_argument('--plots', action='store_true',
                       help='Generate trade-off plots')
    
    args = parser.parse_args()
    
    runner = Phase2AExperimentRunner()
    
    if args.mode == 'single':
        if not args.config:
            raise ValueError("--config required for single mode")
        
        results = runner.run_multi_sigma_sweep(args.config)
        
        if args.plots:
            runner.generate_trade_off_plots(results)
            
    elif args.mode == 'comparison':
        if not args.configs:
            raise ValueError("--configs required for comparison mode")
        
        all_results = runner.run_dataset_comparison(args.configs, args.names)
        
        if args.plots:
            for name, results in all_results.items():
                runner.generate_trade_off_plots(results, f"plots/phase2a/{name}")


if __name__ == "__main__":
    main()