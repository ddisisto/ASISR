"""
Phase 2B Dynamic Spectral Experiment Framework.

Comparative analysis of training-phase-dependent spectral control strategies
vs static approaches, building on Phase 2A multi-σ findings.
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


class Phase2BResults:
    """Container for Phase 2B dynamic vs static comparison results."""
    
    def __init__(self):
        """Initialize Phase 2B results container."""
        self.static_results: Optional[ExperimentResults] = None
        self.dynamic_results: Dict[str, ExperimentResults] = {}
        self.strategy_names = []
        
    def add_static_results(self, results: ExperimentResults) -> None:
        """Add static baseline results."""
        self.static_results = results
        
    def add_dynamic_results(self, strategy: str, results: ExperimentResults) -> None:
        """Add dynamic strategy results."""
        self.dynamic_results[strategy] = results
        if strategy not in self.strategy_names:
            self.strategy_names.append(strategy)
            
    def compare_strategies(self) -> Dict[str, Dict[str, float]]:
        """
        Compare dynamic strategies against static baseline.
        
        Returns:
            Dictionary of performance comparisons with statistical tests
        """
        if self.static_results is None:
            raise ValueError("No static baseline results available")
        
        # Ensure aggregation is complete
        self.static_results.aggregate_results()
        self.static_results.compute_confidence_intervals()
        
        comparisons = {}
        
        for strategy, dynamic_result in self.dynamic_results.items():
            dynamic_result.aggregate_results()
            dynamic_result.compute_confidence_intervals()
            
            comparison = {}
            
            # Performance comparison
            static_acc = self.static_results.aggregated_results['accuracy']
            dynamic_acc = dynamic_result.aggregated_results['accuracy']
            
            comparison['static_accuracy_mean'] = static_acc['final_mean']
            comparison['static_accuracy_std'] = static_acc['final_std']
            comparison['dynamic_accuracy_mean'] = dynamic_acc['final_mean']
            comparison['dynamic_accuracy_std'] = dynamic_acc['final_std']
            
            # Performance improvement
            improvement = dynamic_acc['final_mean'] - static_acc['final_mean']
            comparison['accuracy_improvement'] = improvement
            comparison['improvement_percentage'] = (improvement / static_acc['final_mean']) * 100
            
            # Variance comparison
            variance_reduction = static_acc['final_std'] - dynamic_acc['final_std']
            comparison['variance_reduction'] = variance_reduction
            comparison['variance_reduction_percentage'] = (variance_reduction / static_acc['final_std']) * 100 if static_acc['final_std'] > 0 else 0
            
            # Statistical significance test
            static_values = [result['final_accuracy'] for result in self.static_results.results_per_seed.values()]
            dynamic_values = [result['final_accuracy'] for result in dynamic_result.results_per_seed.values()]
            
            t_stat, p_value = stats.ttest_ind(dynamic_values, static_values)
            comparison['t_statistic'] = t_stat
            comparison['p_value'] = p_value
            comparison['significant'] = p_value < 0.05
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(static_values) - 1) * np.var(static_values, ddof=1) +
                                (len(dynamic_values) - 1) * np.var(dynamic_values, ddof=1)) /
                               (len(static_values) + len(dynamic_values) - 2))
            cohens_d = (np.mean(dynamic_values) - np.mean(static_values)) / pooled_std
            comparison['cohens_d'] = cohens_d
            comparison['effect_size'] = self._interpret_effect_size(abs(cohens_d))
            
            comparisons[strategy] = comparison
            
        return comparisons
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        if d < 0.2:
            return "negligible"
        elif d < 0.5:
            return "small"
        elif d < 0.8:
            return "medium"
        else:
            return "large"


class Phase2BExperimentRunner:
    """
    Phase 2B experiment orchestrator for dynamic vs static spectral control.
    
    Tests hypothesis that training-phase-dependent σ scheduling can improve
    performance-variance trade-offs compared to optimal fixed σ strategies.
    """
    
    def __init__(self):
        """Initialize Phase 2B experiment runner."""
        pass
    
    def run_dynamic_vs_static_comparison(self, 
                                       static_config: str,
                                       dynamic_configs: List[str],
                                       strategy_names: Optional[List[str]] = None) -> Phase2BResults:
        """
        Run comprehensive dynamic vs static comparison.
        
        Args:
            static_config: Path to static baseline configuration
            dynamic_configs: List of paths to dynamic strategy configurations
            strategy_names: Optional names for dynamic strategies
            
        Returns:
            Complete Phase 2B comparison results
        """
        if strategy_names is None:
            strategy_names = [f"dynamic_{i}" for i in range(len(dynamic_configs))]
        
        if len(dynamic_configs) != len(strategy_names):
            raise ValueError("Number of dynamic configs must match strategy names")
        
        print(f"\n{'='*80}")
        print("SPECTRA Phase 2B: Dynamic vs Static Spectral Control")
        print(f"Static Baseline: {Path(static_config).stem}")
        print(f"Dynamic Strategies: {', '.join(strategy_names)}")
        print(f"{'='*80}")
        
        results = Phase2BResults()
        
        # Run static baseline
        print(f"\n{'='*60}")
        print("Running Static Baseline Experiment")
        print(f"{'='*60}")
        
        static_experiment = SPECTRAExperiment(load_config(static_config))
        static_result = static_experiment.run_multi_seed()
        results.add_static_results(static_result)
        
        # Print static summary
        static_result.aggregate_results()
        static_acc = static_result.aggregated_results['accuracy']
        print(f"Static Baseline: {static_acc['final_mean']:.3f} ± {static_acc['final_std']:.3f} accuracy")
        
        # Run dynamic strategies
        for config_path, strategy_name in zip(dynamic_configs, strategy_names):
            print(f"\n{'='*60}")
            print(f"Running Dynamic Strategy: {strategy_name}")
            print(f"Config: {Path(config_path).stem}")
            print(f"{'='*60}")
            
            dynamic_experiment = SPECTRAExperiment(load_config(config_path))
            dynamic_result = dynamic_experiment.run_multi_seed()
            results.add_dynamic_results(strategy_name, dynamic_result)
            
            # Print dynamic summary
            dynamic_result.aggregate_results()
            dynamic_acc = dynamic_result.aggregated_results['accuracy']
            print(f"{strategy_name}: {dynamic_acc['final_mean']:.3f} ± {dynamic_acc['final_std']:.3f} accuracy")
        
        # Statistical comparison
        print(f"\n{'='*40}")
        print("STATISTICAL COMPARISON")
        print(f"{'='*40}")
        
        comparisons = results.compare_strategies()
        
        for strategy, comp in comparisons.items():
            print(f"\n{strategy.upper()}:")
            print(f"  Accuracy: {comp['dynamic_accuracy_mean']:.3f} ± {comp['dynamic_accuracy_std']:.3f}")
            print(f"  vs Static: {comp['accuracy_improvement']:+.3f} ({comp['improvement_percentage']:+.1f}%)")
            print(f"  Variance: {comp['variance_reduction']:+.3f} ({comp['variance_reduction_percentage']:+.1f}%)")
            print(f"  p-value: {comp['p_value']:.4f} {'*' if comp['significant'] else ''}")
            print(f"  Effect size: {comp['effect_size']} (d={comp['cohens_d']:.3f})")
        
        return results
    
    def generate_comparison_plots(self, results: Phase2BResults, 
                                output_dir: str = "plots/phase2b") -> None:
        """
        Generate publication-quality Phase 2B comparison plots.
        
        Args:
            results: Phase 2B comparison results
            output_dir: Directory to save plots
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("Set2")
        
        # Performance comparison plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('SPECTRA Phase 2B: Dynamic vs Static Spectral Control', fontsize=16)
        
        # Accuracy comparison
        strategies = ['Static'] + results.strategy_names
        accuracies = []
        accuracy_errors = []
        
        # Static baseline
        static_acc = results.static_results.aggregated_results['accuracy']
        accuracies.append(static_acc['final_mean'])
        accuracy_errors.append(static_acc['final_std'])
        
        # Dynamic strategies
        for strategy in results.strategy_names:
            dynamic_acc = results.dynamic_results[strategy].aggregated_results['accuracy']
            accuracies.append(dynamic_acc['final_mean'])
            accuracy_errors.append(dynamic_acc['final_std'])
        
        bars1 = ax1.bar(strategies, accuracies, yerr=accuracy_errors, capsize=5, alpha=0.7)
        ax1.set_title('Final Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Color static bar differently
        bars1[0].set_color('red')
        bars1[0].set_alpha(0.5)
        
        # Variance comparison
        variances = [acc_err for acc_err in accuracy_errors]
        bars2 = ax2.bar(strategies, variances, alpha=0.7)
        ax2.set_title('Variance Comparison (Lower is Better)', fontweight='bold')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(True, alpha=0.3)
        
        # Color static bar differently
        bars2[0].set_color('red')
        bars2[0].set_alpha(0.5)
        
        plt.tight_layout()
        plt.savefig(output_path / 'phase2b_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training dynamics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SPECTRA Phase 2B: Training Dynamics Comparison', fontsize=16)
        
        metrics = ['accuracy', 'criticality_score', 'spectral_radius_avg', 'boundary_fractal_dim']
        metric_titles = ['Accuracy Evolution', 'Criticality Score', 'Spectral Radius', 'Boundary Complexity']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx // 2, idx % 2]
            
            # Skip trajectory plots if data not available
            if 'trajectory_mean' not in results.static_results.aggregated_results.get(metric, {}):
                ax.text(0.5, 0.5, f'{metric}\nTrajectory data\nnot available', 
                       transform=ax.transAxes, ha='center', va='center')
                continue
                
            # Plot static baseline
            static_history = results.static_results.aggregated_results[metric]['trajectory_mean']
            static_epochs = range(len(static_history))
            ax.plot(static_epochs, static_history, 'r--', linewidth=2, label='Static', alpha=0.7)
            
            # Plot dynamic strategies
            for strategy in results.strategy_names:
                if 'trajectory_mean' in results.dynamic_results[strategy].aggregated_results.get(metric, {}):
                    dynamic_history = results.dynamic_results[strategy].aggregated_results[metric]['trajectory_mean']
                    dynamic_epochs = range(len(dynamic_history))
                    ax.plot(dynamic_epochs, dynamic_history, linewidth=2, label=strategy)
            
            ax.set_title(title, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'phase2b_dynamics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Phase 2B plots saved to {output_path}")


def main():
    """Main entry point for Phase 2B experiments."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPECTRA Phase 2B Dynamic vs Static Experiment")
    parser.add_argument('--static-config', required=True,
                       help='Static baseline configuration file')
    parser.add_argument('--dynamic-configs', nargs='+', required=True,
                       help='Dynamic strategy configuration files')
    parser.add_argument('--strategy-names', nargs='+',
                       help='Names for dynamic strategies')
    parser.add_argument('--plots', action='store_true',
                       help='Generate comparison plots')
    
    args = parser.parse_args()
    
    runner = Phase2BExperimentRunner()
    
    results = runner.run_dynamic_vs_static_comparison(
        static_config=args.static_config,
        dynamic_configs=args.dynamic_configs,
        strategy_names=args.strategy_names
    )
    
    if args.plots:
        runner.generate_comparison_plots(results)


if __name__ == "__main__":
    main()