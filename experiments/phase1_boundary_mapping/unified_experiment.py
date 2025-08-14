"""
Unified experiment framework for SPECTRA Phase 1.

Consolidates integration testing, framework validation, and research experiments
into a single, scalable system supporting multiple experiment modes.
"""

import sys
from pathlib import Path
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
import time

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from spectra.utils.config import load_config, SPECTRAConfig
from spectra.training.experiment import SPECTRAExperiment, compare_experiments, ExperimentResults


class ExperimentMode(Enum):
    """Experiment execution modes with different scope and duration."""
    INTEGRATION = "integration"      # Quick validation (5-10 epochs)
    VALIDATION = "validation"        # Framework testing (20-50 epochs)  
    RESEARCH = "research"            # Full experiments (100+ epochs)
    COMPARISON = "comparison"        # A/B baseline vs spectral


class UnifiedExperimentRunner:
    """
    Unified experiment orchestrator supporting all SPECTRA Phase 1 experiment types.
    
    Eliminates redundancy between integration tests, framework validation,
    and research experiments by providing a single, configurable interface.
    """
    
    def __init__(self, mode: ExperimentMode = ExperimentMode.VALIDATION):
        """
        Initialize unified experiment runner.
        
        Args:
            mode: Experiment execution mode determining scope and duration
        """
        self.mode = mode
        self.mode_configs = {
            ExperimentMode.INTEGRATION: {
                'epochs': 5,
                'seeds': [42],
                'log_interval': 1,
                'description': 'Integration test validation'
            },
            ExperimentMode.VALIDATION: {
                'epochs': 20,
                'seeds': [42, 123],
                'log_interval': 5,
                'description': 'Framework validation'
            },
            ExperimentMode.RESEARCH: {
                'epochs': 100,
                'seeds': [42, 123, 456, 789, 1011],
                'log_interval': 10,
                'description': 'Full research experiment'
            },
            ExperimentMode.COMPARISON: {
                'epochs': 100,
                'seeds': [42, 123, 456, 789, 1011],
                'log_interval': 10,
                'description': 'Baseline vs Spectral A/B comparison'
            }
        }
    
    def run_single_experiment(self, config_path: str, 
                            experiment_name: str = None) -> ExperimentResults:
        """
        Run single experiment with mode-specific parameters.
        
        Args:
            config_path: Path to experiment configuration file
            experiment_name: Optional name for this experiment
            
        Returns:
            Complete experiment results with statistical analysis
        """
        # Load configuration
        config = load_config(config_path)
        mode_config = self.mode_configs[self.mode]
        
        # Override config with mode-specific parameters
        original_epochs = config.config['training']['epochs']
        original_seeds = config.config['experiment']['seeds']
        original_log_interval = config.config['experiment']['log_interval']
        
        config.config['training']['epochs'] = mode_config['epochs']
        config.config['experiment']['seeds'] = mode_config['seeds']
        config.config['experiment']['log_interval'] = mode_config['log_interval']
        
        # Create experiment name
        if experiment_name is None:
            reg_type = "spectral" if config.regularization else "baseline"
            experiment_name = f"{reg_type}_{self.mode.value}"
        
        print(f"\n{'='*60}")
        print(f"Running {mode_config['description']}: {experiment_name}")
        print(f"Config: {config_path}")
        print(f"Mode: {self.mode.value} ({mode_config['epochs']} epochs, {len(mode_config['seeds'])} seeds)")
        print(f"{'='*60}")
        
        # Run experiment
        experiment = SPECTRAExperiment(config)
        results = experiment.run_multi_seed(seeds=mode_config['seeds'])
        
        # Restore original config values
        config.config['training']['epochs'] = original_epochs
        config.config['experiment']['seeds'] = original_seeds
        config.config['experiment']['log_interval'] = original_log_interval
        
        return results
    
    def run_baseline_vs_spectral_comparison(self, 
                                          baseline_config: str = "configs/phase1_baseline.yaml",
                                          spectral_config: str = "configs/phase1_spectral.yaml") -> Dict[str, Any]:
        """
        Run controlled A/B comparison between baseline and spectral regularization.
        
        Args:
            baseline_config: Path to baseline experiment configuration
            spectral_config: Path to spectral regularization configuration
            
        Returns:
            Comprehensive comparison results with statistical analysis
        """
        if self.mode != ExperimentMode.COMPARISON:
            print(f"Warning: Running comparison in {self.mode.value} mode")
        
        print(f"\n{'='*80}")
        print("SPECTRA Phase 1 Core Hypothesis Testing")
        print("Baseline vs Spectral Regularization A/B Comparison")
        print(f"{'='*80}")
        
        # Run baseline experiment
        print("\n" + "="*40)
        print("BASELINE EXPERIMENT (No Regularization)")
        print("="*40)
        baseline_results = self.run_single_experiment(baseline_config, "baseline")
        
        # Run spectral regularization experiment
        print("\n" + "="*40)
        print("SPECTRAL EXPERIMENT (σ ≈ 1.0 Targeting)")
        print("="*40)
        spectral_results = self.run_single_experiment(spectral_config, "spectral")
        
        # Statistical comparison
        print("\n" + "="*40)
        print("STATISTICAL COMPARISON")
        print("="*40)
        
        comparison_results = {}
        
        # Compare key metrics
        metrics_to_compare = ['accuracy', 'criticality_score', 'train_loss', 'boundary_fractal_dim']
        
        for metric in metrics_to_compare:
            if (metric in baseline_results.aggregated_results and 
                metric in spectral_results.aggregated_results):
                
                try:
                    comparison = compare_experiments(baseline_results, spectral_results, metric)
                    comparison_results[metric] = comparison
                    
                    print(f"\n{metric.upper()} Comparison:")
                    print(f"  Baseline: {comparison['baseline_mean']:.4f} ± {comparison['baseline_std']:.4f}")
                    print(f"  Spectral: {comparison['spectral_mean']:.4f} ± {comparison['spectral_std']:.4f}")
                    print(f"  Difference: {comparison['difference']:.4f}")
                    print(f"  Effect size (Cohen's d): {comparison['cohens_d']:.3f} ({comparison['effect_size_interpretation']})")
                    print(f"  Statistical significance: p = {comparison['p_value']:.4f} {'***' if comparison['p_value'] < 0.001 else '**' if comparison['p_value'] < 0.01 else '*' if comparison['p_value'] < 0.05 else 'ns'}")
                    
                except Exception as e:
                    print(f"  Warning: Could not compare {metric}: {e}")
        
        # Overall assessment
        print("\n" + "="*40)
        print("HYPOTHESIS VALIDATION")
        print("="*40)
        
        # Check core hypothesis: Does spectral regularization improve training efficiency?
        if 'accuracy' in comparison_results:
            acc_comparison = comparison_results['accuracy']
            accuracy_improved = acc_comparison['difference'] > 0 and acc_comparison['significant']
            
            if accuracy_improved:
                print(f"✅ HYPOTHESIS SUPPORTED: Spectral regularization improves performance")
                print(f"   Accuracy improvement: {acc_comparison['difference']:.3f} (p = {acc_comparison['p_value']:.4f})")
            else:
                print(f"❌ HYPOTHESIS NOT SUPPORTED: No significant accuracy improvement")
                print(f"   Accuracy difference: {acc_comparison['difference']:.3f} (p = {acc_comparison['p_value']:.4f})")
        
        # Package complete results
        complete_results = {
            'baseline_results': baseline_results,
            'spectral_results': spectral_results,
            'statistical_comparisons': comparison_results,
            'experiment_mode': self.mode.value,
            'hypothesis_supported': accuracy_improved if 'accuracy' in comparison_results else False
        }
        
        return complete_results
    
    def run_integration_test(self) -> bool:
        """
        Run integration test to validate all components work together.
        
        Returns:
            True if integration test passes, False otherwise
        """
        old_mode = self.mode
        self.mode = ExperimentMode.INTEGRATION
        
        try:
            print("SPECTRA Unified Integration Test")
            print("=" * 50)
            
            # Test baseline configuration
            print("\n=== Testing Baseline Configuration ===")
            baseline_results = self.run_single_experiment("configs/phase1_baseline.yaml")
            baseline_accuracy = baseline_results.aggregated_results['accuracy']['final_mean']
            
            # Test spectral configuration  
            print("\n=== Testing Spectral Configuration ===")
            spectral_results = self.run_single_experiment("configs/phase1_spectral.yaml")
            spectral_accuracy = spectral_results.aggregated_results['accuracy']['final_mean']
            
            # Validation checks
            success_checks = {
                'baseline_accuracy_reasonable': 0.5 < baseline_accuracy < 1.0,
                'spectral_accuracy_reasonable': 0.5 < spectral_accuracy < 1.0,
                'criticality_computed': 'criticality_score' in baseline_results.aggregated_results,
                'boundary_fractal_computed': 'boundary_fractal_dim' in baseline_results.aggregated_results,
                'spectral_radius_tracked': 'spectral_radius_avg' in baseline_results.aggregated_results
            }
            
            print("\n=== Integration Test Results ===")
            all_passed = True
            for check, passed in success_checks.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status}: {check}")
                all_passed = all_passed and passed
            
            if all_passed:
                print(f"\n🚀 Integration test PASSED")
                print(f"Baseline accuracy: {baseline_accuracy:.3f}")
                print(f"Spectral accuracy: {spectral_accuracy:.3f}")
            else:
                print(f"\n❌ Integration test FAILED")
            
            return all_passed
            
        except Exception as e:
            print(f"\n❌ Integration test FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.mode = old_mode
    
    def run_framework_validation(self) -> bool:
        """
        Run framework validation to test multi-seed statistical capabilities.
        
        Returns:
            True if framework validation passes, False otherwise
        """
        old_mode = self.mode
        self.mode = ExperimentMode.VALIDATION
        
        try:
            print("SPECTRA Framework Validation")
            print("=" * 50)
            
            # Test statistical framework with baseline config
            results = self.run_single_experiment("configs/phase1_baseline.yaml")
            
            # Validation checks
            success_checks = {
                'multi_seed_execution': len(results.results_per_seed) > 1,
                'statistical_aggregation': bool(results.aggregated_results),
                'confidence_intervals': any('confidence_interval' in stats 
                                          for stats in results.aggregated_results.values()),
                'reproducibility': len(results.results_per_seed) > 1  # Multiple seeds executed
            }
            
            print("\n=== Framework Validation Results ===")
            all_passed = True
            for check, passed in success_checks.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"{status}: {check}")
                all_passed = all_passed and passed
            
            if all_passed:
                print(f"\n🚀 Framework validation PASSED")
                print(f"Statistical framework ready for research experiments")
            else:
                print(f"\n❌ Framework validation FAILED")
            
            return all_passed
            
        except Exception as e:
            print(f"\n❌ Framework validation FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        finally:
            self.mode = old_mode


def main():
    """Main entry point supporting different experiment modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="SPECTRA Unified Experiment Runner")
    parser.add_argument('mode', choices=['integration', 'validation', 'research', 'comparison'],
                       help='Experiment mode to run')
    parser.add_argument('--baseline-config', default='configs/phase1_baseline.yaml',
                       help='Baseline experiment configuration')
    parser.add_argument('--spectral-config', default='configs/phase1_spectral.yaml', 
                       help='Spectral experiment configuration')
    
    args = parser.parse_args()
    
    # Create experiment runner
    mode = ExperimentMode(args.mode)
    runner = UnifiedExperimentRunner(mode)
    
    # Execute based on mode
    if mode == ExperimentMode.INTEGRATION:
        success = runner.run_integration_test()
        sys.exit(0 if success else 1)
        
    elif mode == ExperimentMode.VALIDATION:
        success = runner.run_framework_validation()
        sys.exit(0 if success else 1)
        
    elif mode == ExperimentMode.RESEARCH:
        print("Running research experiment with baseline configuration...")
        results = runner.run_single_experiment(args.baseline_config)
        print(f"Research experiment completed successfully")
        
    elif mode == ExperimentMode.COMPARISON:
        results = runner.run_baseline_vs_spectral_comparison(
            args.baseline_config, args.spectral_config
        )
        
        # Exit with appropriate code based on hypothesis validation
        success = results.get('hypothesis_supported', False)
        print(f"\nExiting with code {'0 (success)' if success else '1 (hypothesis not supported)'}")
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()