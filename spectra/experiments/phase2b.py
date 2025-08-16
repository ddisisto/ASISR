"""
Phase 2B Dynamic vs Static Spectral Control Experiments

Modularized experiment orchestration for Phase 2B research comparing
dynamic σ scheduling strategies against static baselines.
"""

from pathlib import Path
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from .base import BaseExperiment, ExperimentResult, ComparisonResult


class Phase2BComparisonExperiment(BaseExperiment):
    """
    Phase 2B experiment comparing dynamic vs static spectral control.
    
    Tests hypothesis that training-phase-dependent σ scheduling improves
    performance-variance trade-offs compared to optimal fixed σ strategies.
    """
    
    def get_phase_name(self) -> str:
        """Return phase name for output directory structure."""
        return "phase2b"
    
    def run(self, 
            static_config: str,
            dynamic_configs: List[str], 
            strategy_names: List[str],
            generate_plots: bool = True) -> ComparisonResult:
        """
        Run comprehensive dynamic vs static comparison.
        
        Args:
            static_config: Path to static baseline configuration
            dynamic_configs: List of dynamic strategy configurations  
            strategy_names: Names for dynamic strategies
            generate_plots: Generate comparison visualizations
            
        Returns:
            Complete Phase 2B comparison results
        """
        print(f"{'='*80}")
        print("SPECTRA Phase 2B: Dynamic vs Static Spectral Control")
        print(f"Static Baseline: {Path(static_config).stem}")
        print(f"Dynamic Strategies: {', '.join(strategy_names)}")
        print(f"{'='*80}")
        
        # Run static baseline
        baseline_result = self.run_single_experiment(
            config_path=static_config,
            experiment_name="Static"
        )
        
        # Run dynamic strategies
        comparison_results = []
        for config_path, strategy_name in zip(dynamic_configs, strategy_names):
            result = self.run_single_experiment(
                config_path=config_path,
                experiment_name=strategy_name
            )
            comparison_results.append(result)
        
        # Statistical analysis
        statistical_tests = self._compute_statistical_comparisons(
            baseline_result, comparison_results
        )
        
        # Create comparison result
        comparison_result = ComparisonResult(
            experiment_name="Phase2B_Comparison",
            baseline_result=baseline_result,
            comparison_results=comparison_results,
            statistical_tests=statistical_tests,
            output_dir=self.get_output_dir()
        )
        
        # Generate plots
        if generate_plots:
            plot_paths = self.generate_plots(comparison_result)
            print(f"\\nPlots saved to: {comparison_result.output_dir}")
            for path in plot_paths:
                print(f"  - {path.name}")
        
        return comparison_result
    
    def _compute_statistical_comparisons(self, 
                                       baseline: ExperimentResult,
                                       comparisons: List[ExperimentResult]) -> Dict[str, Any]:
        """
        Compute statistical tests comparing dynamic strategies to baseline.
        
        Args:
            baseline: Static baseline results
            comparisons: Dynamic strategy results
            
        Returns:
            Statistical test results for each comparison
        """
        tests = {}
        
        # Get baseline accuracy data
        if 'accuracy' not in baseline.results.aggregated_results:
            return tests
        
        baseline_values = []
        for seed_result in baseline.results.results_per_seed.values():
            baseline_values.append(seed_result['final_accuracy'])
        
        # Compare each dynamic strategy
        for comp_result in comparisons:
            if 'accuracy' not in comp_result.results.aggregated_results:
                continue
                
            comp_values = []
            for seed_result in comp_result.results.results_per_seed.values():
                comp_values.append(seed_result['final_accuracy'])
            
            if len(comp_values) < 2 or len(baseline_values) < 2:
                continue
            
            # T-test
            t_stat, p_value = stats.ttest_ind(comp_values, baseline_values)
            
            # Effect size (Cohen's d)
            baseline_mean = sum(baseline_values) / len(baseline_values)
            comp_mean = sum(comp_values) / len(comp_values)
            pooled_std = (sum([(x - baseline_mean)**2 for x in baseline_values]) + 
                         sum([(x - comp_mean)**2 for x in comp_values])) / (len(baseline_values) + len(comp_values) - 2)
            pooled_std = pooled_std ** 0.5
            
            cohens_d = (comp_mean - baseline_mean) / pooled_std if pooled_std > 0 else 0
            
            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                effect_size = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_size = "small"
            elif abs(cohens_d) < 0.8:
                effect_size = "medium"
            else:
                effect_size = "large"
            
            tests[comp_result.experiment_name] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'cohens_d': cohens_d,
                'effect_size': effect_size,
                'accuracy_improvement': comp_mean - baseline_mean,
                'baseline_mean': baseline_mean,
                'comparison_mean': comp_mean
            }
        
        return tests
    
    def generate_plots(self, result: ComparisonResult) -> List[Path]:
        """
        Generate Phase 2B comparison plots.
        
        Args:
            result: Phase 2B comparison results
            
        Returns:
            List of generated plot file paths
        """
        output_dir = result.output_dir
        plot_paths = []
        
        # Generate experiment-specific filename suffix
        if result.comparison_results:
            # Use the first comparison strategy name for filename
            experiment_suffix = result.comparison_results[0].experiment_name
        else:
            experiment_suffix = "baseline"
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("Set2")
        
        # 1. Accuracy and Variance Comparison
        comparison_plot = self._generate_comparison_plot(result)
        comparison_path = output_dir / f"phase2b_comparison_{experiment_suffix}.png"
        comparison_plot.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close(comparison_plot)
        plot_paths.append(comparison_path)
        
        # 2. Training Dynamics
        dynamics_plot = self._generate_dynamics_plot(result)
        dynamics_path = output_dir / f"phase2b_dynamics_{experiment_suffix}.png"
        dynamics_plot.savefig(dynamics_path, dpi=300, bbox_inches='tight')
        plt.close(dynamics_plot)
        plot_paths.append(dynamics_path)
        
        return plot_paths
    
    def _generate_comparison_plot(self, result: ComparisonResult) -> plt.Figure:
        """Generate accuracy and variance comparison plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Collect data
        names = ['Static']
        accuracies = [result.baseline_result.final_accuracy]
        stds = [result.baseline_result.final_accuracy_std]
        
        for comp_result in result.comparison_results:
            names.append(comp_result.experiment_name)
            accuracies.append(comp_result.final_accuracy)
            stds.append(comp_result.final_accuracy_std)
        
        # Colors: red for baseline, green shades for dynamic
        colors = ['salmon'] + ['lightseagreen'] * (len(names) - 1)
        
        # Accuracy comparison
        bars1 = ax1.bar(names, accuracies, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title('Final Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1.0)
        
        # Add error bars
        ax1.errorbar(names, accuracies, yerr=stds, fmt='none', 
                    color='black', capsize=3, capthick=1)
        
        # Variance comparison
        bars2 = ax2.bar(names, stds, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_title('Variance Comparison (Lower is Better)')
        ax2.set_ylabel('Standard Deviation')
        
        plt.suptitle('SPECTRA Phase 2B: Dynamic vs Static Spectral Control', fontsize=14)
        plt.tight_layout()
        
        return fig
    
    def _generate_dynamics_plot(self, result: ComparisonResult) -> plt.Figure:
        """Generate training dynamics comparison plot."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        # Metrics to plot
        metrics = ['accuracy', 'criticality_score', 'spectral_radius_avg', 'boundary_fractal_dim']
        metric_titles = ['Accuracy Evolution', 'Criticality Score', 'Spectral Radius', 'Boundary Complexity']
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[idx]
            
            # Plot baseline
            baseline_results = result.baseline_result.results.aggregated_results
            if metric in baseline_results and 'trajectory_mean' in baseline_results[metric]:
                baseline_history = baseline_results[metric]['trajectory_mean']
                epochs = range(len(baseline_history))
                ax.plot(epochs, baseline_history, 'r--', linewidth=2, 
                       label='Static', alpha=0.7)
            
            # Plot dynamic strategies
            for comp_result in result.comparison_results:
                comp_results = comp_result.results.aggregated_results
                if metric in comp_results and 'trajectory_mean' in comp_results[metric]:
                    history = comp_results[metric]['trajectory_mean']
                    epochs = range(len(history))
                    ax.plot(epochs, history, linewidth=2, 
                           label=comp_result.experiment_name, alpha=0.8)
            
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('SPECTRA Phase 2B: Training Dynamics Comparison', fontsize=14)
        plt.tight_layout()
        
        return fig