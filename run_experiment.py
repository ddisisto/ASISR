#!/usr/bin/env python3
"""
Standardized SPECTRA Experiment Runner

Unified interface for running all SPECTRA experiments with consistent
progress reporting, output paths, and result management.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from spectra.utils.config import load_config
from spectra.training.experiment import SPECTRAExperiment
from spectra.experiments import Phase2BComparisonExperiment


class SPECTRAExperimentRunner:
    """Standardized experiment runner for all SPECTRA phases."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.output_base = Path("plots")
        self.output_base.mkdir(exist_ok=True)
    
    def _check_existing_results(self, strategy_names: List[str]) -> None:
        """
        Check for existing experiment results and prevent overwrites.
        
        Args:
            strategy_names: List of experiment strategy names to check
            
        Raises:
            SystemExit: If existing results found and overwrite not allowed
        """
        phase2b_dir = self.output_base / "phase2b"
        existing_experiments = []
        
        if not phase2b_dir.exists():
            return  # No existing results
        
        # Check for existing experiment directories
        for strategy_name in strategy_names:
            experiment_dir = phase2b_dir / strategy_name
            if experiment_dir.exists() and any(experiment_dir.iterdir()):
                existing_experiments.append(strategy_name)
        
        # Check for existing plot files in root phase2b directory
        existing_plots = []
        for strategy_name in strategy_names:
            comparison_plot = phase2b_dir / f"phase2b_comparison_{strategy_name}.png"
            dynamics_plot = phase2b_dir / f"phase2b_dynamics_{strategy_name}.png"
            if comparison_plot.exists() or dynamics_plot.exists():
                existing_plots.append(strategy_name)
        
        if existing_experiments or existing_plots:
            print("❌ ERROR: Existing experiment results found!")
            print()
            if existing_experiments:
                print("📁 Experiment directories with existing data:")
                for exp in existing_experiments:
                    exp_dir = phase2b_dir / exp
                    file_count = len(list(exp_dir.iterdir()))
                    print(f"  - {exp_dir} ({file_count} files)")
            
            if existing_plots:
                print("🖼️  Existing plot files:")
                for exp in existing_plots:
                    comparison_plot = phase2b_dir / f"phase2b_comparison_{exp}.png"
                    dynamics_plot = phase2b_dir / f"phase2b_dynamics_{exp}.png"
                    if comparison_plot.exists():
                        print(f"  - {comparison_plot}")
                    if dynamics_plot.exists():
                        print(f"  - {dynamics_plot}")
            
            print()
            print("🛡️  To protect your existing results, this experiment has been cancelled.")
            print("💡 To overwrite existing results, run with --overwrite flag:")
            print(f"   python run_experiment.py phase2b [args] --overwrite")
            print()
            sys.exit(1)
    
    def run_single_experiment(self, config_path: str, output_dir: Optional[str] = None) -> None:
        """
        Run a single experiment from config file.
        
        Args:
            config_path: Path to YAML configuration file
            output_dir: Optional output directory override
        """
        config = load_config(config_path)
        experiment = SPECTRAExperiment(config)
        
        print(f"{'='*80}")
        print(f"SPECTRA Single Experiment: {Path(config_path).stem}")
        print(f"{'='*80}")
        
        results = experiment.run_multi_seed()
        results.aggregate_results()
        
        # Print summary
        if 'accuracy' in results.aggregated_results:
            acc_stats = results.aggregated_results['accuracy']
            print(f"\nFinal Results:")
            print(f"  Accuracy: {acc_stats['final_mean']:.3f} ± {acc_stats['final_std']:.3f}")
            if 'criticality_score' in results.aggregated_results:
                crit_stats = results.aggregated_results['criticality_score']
                print(f"  Criticality: {crit_stats['final_mean']:.3f} ± {crit_stats['final_std']:.3f}")
    
    def run_phase2b_comparison(self, 
                              static_config: str,
                              dynamic_configs: List[str],
                              strategy_names: List[str],
                              plots: bool = True,
                              overwrite: bool = False) -> None:
        """
        Run Phase 2B dynamic vs static comparison.
        
        Args:
            static_config: Path to static baseline configuration
            dynamic_configs: List of dynamic strategy configurations
            strategy_names: Names for dynamic strategies  
            plots: Generate comparison plots
            overwrite: Allow overwriting existing results
        """
        # Check for existing results and prevent overwrites unless explicitly allowed
        if not overwrite:
            self._check_existing_results(strategy_names)
        
        experiment = Phase2BComparisonExperiment(output_base=self.output_base)
        result = experiment.run(
            static_config=static_config,
            dynamic_configs=dynamic_configs,
            strategy_names=strategy_names,
            generate_plots=plots
        )
        
        # Print statistical summary
        print(f"\n{'='*60}")
        print("Statistical Analysis Summary:")
        for strategy_name, stats in result.statistical_tests.items():
            improvement = stats['accuracy_improvement']
            p_val = stats['p_value']
            effect = stats['effect_size']
            significance = "*" if stats['significant'] else ""
            print(f"{strategy_name}: {improvement:+.1%} accuracy improvement")
            print(f"  p-value: {p_val:.4f}{significance}, Effect: {effect}")
        print(f"{'='*60}")
    
    def run_phase1_boundary(self, config_path: str, comparison_mode: bool = False) -> None:
        """
        Run Phase 1 boundary mapping experiment.
        
        Args:
            config_path: Path to configuration file
            comparison_mode: Run baseline vs spectral comparison
        """
        print("🔬 Running Phase 1 boundary mapping experiment...")
        
        try:
            # Import the unified experiment runner
            import sys
            from pathlib import Path
            phase1_path = Path(__file__).parent / "experiments" / "phase1_boundary_mapping"
            sys.path.insert(0, str(phase1_path))
            
            from unified_experiment import UnifiedExperimentRunner, ExperimentMode
            
            if comparison_mode:
                print("Running baseline vs spectral comparison...")
                mode = ExperimentMode.COMPARISON
                runner = UnifiedExperimentRunner(mode)
                
                # Use default config paths for comparison
                baseline_config = "configs/phase1_baseline.yaml"
                spectral_config = "configs/phase1_spectral.yaml"
                
                results = runner.run_baseline_vs_spectral_comparison(
                    baseline_config, spectral_config
                )
                print("✅ Phase 1 comparison experiment completed")
            else:
                print(f"Running single experiment with config: {config_path}")
                mode = ExperimentMode.RESEARCH
                runner = UnifiedExperimentRunner(mode)
                
                results = runner.run_single_experiment(config_path)
                print("✅ Phase 1 experiment completed")
                
        except ImportError as e:
            print(f"❌ Failed to import Phase 1 experiment modules: {e}")
            print("Please ensure experiments/phase1_boundary_mapping/ is available")
        except Exception as e:
            print(f"❌ Phase 1 experiment failed: {e}")
            raise
    
    def run_visualization(self, output_dir: str = "plots/phase2c") -> None:
        """
        Generate Phase 2C visualization gallery.
        
        Args:
            output_dir: Directory to save visualization plots
        """
        print(f"📈 Generating SPECTRA visualization gallery...")
        print(f"Output directory: {output_dir}")
        
        try:
            from spectra.visualization.schedules import save_schedule_gallery
            save_schedule_gallery(output_dir)
            print(f"✅ Visualization gallery saved to {output_dir}/")
        except ImportError as e:
            print(f"❌ Failed to import visualization module: {e}")
            return
        except Exception as e:
            print(f"❌ Failed to generate visualizations: {e}")
            return
    
    def list_available_configs(self) -> None:
        """List all available configuration files."""
        config_dir = Path("configs")
        if not config_dir.exists():
            print("No configs/ directory found")
            return
        
        configs = list(config_dir.glob("*.yaml"))
        if not configs:
            print("No YAML configuration files found in configs/")
            return
        
        print("Available configuration files:")
        for phase in ["phase1", "phase2a", "phase2b", "phase2c"]:
            phase_configs = [c for c in configs if phase in c.name]
            if phase_configs:
                print(f"\n{phase.upper()}:")
                for config in sorted(phase_configs):
                    print(f"  {config.name}")


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="Standardized SPECTRA Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment
  python run_experiment.py single configs/phase2b_linear_schedule.yaml

  # Run Phase 2B comparison
  python run_experiment.py phase2b \\
    --static configs/phase2b_static_comparison.yaml \\
    --dynamic configs/phase2b_linear_schedule.yaml configs/phase2b_exponential_schedule.yaml \\
    --names Linear Exponential \\
    --plots

  # Generate Phase 2C visualizations
  python run_experiment.py visualize --output-dir plots/phase2c

  # List available configs
  python run_experiment.py list
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Experiment type')
    
    # Single experiment
    single_parser = subparsers.add_parser('single', help='Run single experiment')
    single_parser.add_argument('config', help='Configuration file path')
    single_parser.add_argument('--output-dir', help='Output directory override')
    
    # Phase 2B comparison
    phase2b_parser = subparsers.add_parser('phase2b', help='Run Phase 2B comparison')
    phase2b_parser.add_argument('--static', required=True, help='Static baseline config')
    phase2b_parser.add_argument('--dynamic', nargs='+', required=True, help='Dynamic strategy configs')
    phase2b_parser.add_argument('--names', nargs='+', required=True, help='Strategy names')
    phase2b_parser.add_argument('--plots', action='store_true', help='Generate plots')
    phase2b_parser.add_argument('--overwrite', action='store_true', help='Allow overwriting existing experiment results')
    
    # Phase 1 boundary
    phase1_parser = subparsers.add_parser('phase1', help='Run Phase 1 boundary mapping')
    phase1_parser.add_argument('config', help='Configuration file path')
    phase1_parser.add_argument('--comparison', action='store_true', help='Run comparison mode')
    
    # Visualization
    visualize_parser = subparsers.add_parser('visualize', help='Generate Phase 2C visualization gallery')
    visualize_parser.add_argument('--output-dir', default='plots/phase2c', help='Output directory for plots (default: plots/phase2c)')
    
    # List configs
    subparsers.add_parser('list', help='List available configuration files')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    runner = SPECTRAExperimentRunner()
    
    if args.command == 'single':
        runner.run_single_experiment(args.config, args.output_dir)
    elif args.command == 'phase2b':
        runner.run_phase2b_comparison(
            args.static, args.dynamic, args.names, args.plots, args.overwrite
        )
    elif args.command == 'phase1':
        runner.run_phase1_boundary(args.config, args.comparison)
    elif args.command == 'visualize':
        runner.run_visualization(args.output_dir)
    elif args.command == 'list':
        runner.list_available_configs()


if __name__ == "__main__":
    main()