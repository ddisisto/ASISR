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
from experiments.phase2b_dynamic_spectral.phase2b_experiment import Phase2BExperimentRunner


class SPECTRAExperimentRunner:
    """Standardized experiment runner for all SPECTRA phases."""
    
    def __init__(self):
        """Initialize experiment runner."""
        self.output_base = Path("plots")
        self.output_base.mkdir(exist_ok=True)
    
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
                              plots: bool = True) -> None:
        """
        Run Phase 2B dynamic vs static comparison.
        
        Args:
            static_config: Path to static baseline configuration
            dynamic_configs: List of dynamic strategy configurations
            strategy_names: Names for dynamic strategies  
            plots: Generate comparison plots
        """
        runner = Phase2BExperimentRunner()
        results = runner.run_dynamic_vs_static_comparison(
            static_config=static_config,
            dynamic_configs=dynamic_configs,
            strategy_names=strategy_names
        )
        
        if plots:
            # Ensure output directory exists
            output_dir = self.output_base / "phase2b"
            output_dir.mkdir(exist_ok=True)
            runner.generate_comparison_plots(results, str(output_dir))
            print(f"\nPlots saved to: {output_dir}")
    
    def run_phase1_boundary(self, config_path: str, comparison_mode: bool = False) -> None:
        """
        Run Phase 1 boundary mapping experiment.
        
        Args:
            config_path: Path to configuration file
            comparison_mode: Run baseline vs spectral comparison
        """
        # TODO: Implement Phase 1 standardized runner
        print("Phase 1 boundary mapping runner not yet implemented")
        print(f"Please run from experiments/phase1_boundary_mapping/ for now")
    
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
    
    # Phase 1 boundary
    phase1_parser = subparsers.add_parser('phase1', help='Run Phase 1 boundary mapping')
    phase1_parser.add_argument('config', help='Configuration file path')
    phase1_parser.add_argument('--comparison', action='store_true', help='Run comparison mode')
    
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
            args.static, args.dynamic, args.names, args.plots
        )
    elif args.command == 'phase1':
        runner.run_phase1_boundary(args.config, args.comparison)
    elif args.command == 'list':
        runner.list_available_configs()


if __name__ == "__main__":
    main()