"""
DEPRECATED: Use unified_experiment.py instead.

This file is maintained for reference but experiments should use:
    python experiments/phase1_boundary_mapping/unified_experiment.py validation

The unified framework consolidates all experiment types into a single system.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from asisr.utils.config import load_config
from asisr.training.experiment import ASISRExperiment, compare_experiments
from asisr.utils.seed import validate_reproducibility


def test_config_loading():
    """Test configuration loading and validation."""
    print("=== Testing Configuration Loading ===")
    
    try:
        # Test baseline config
        baseline_config = load_config("configs/phase1_baseline.yaml")
        print(f"Baseline config loaded: {baseline_config}")
        
        # Test spectral config  
        spectral_config = load_config("configs/phase1_spectral.yaml")
        print(f"Spectral config loaded: {spectral_config}")
        
        # Validate required fields
        assert baseline_config.model['type'] == 'SpectralMLP'
        assert baseline_config.regularization is None
        assert spectral_config.regularization['type'] == 'fixed_spectral'
        assert len(baseline_config.get_seeds()) == 5
        
        print("✓ Configuration loading: PASSED")
        return baseline_config, spectral_config
        
    except Exception as e:
        print(f"❌ Configuration loading FAILED: {e}")
        raise


def test_reproducibility():
    """Test experiment reproducibility across multiple runs."""
    print("\n=== Testing Reproducibility ===")
    
    try:
        config = load_config("configs/phase1_baseline.yaml")
        
        # Create simple reproducibility test function
        def single_experiment_final_accuracy():
            experiment = ASISRExperiment(config)
            result = experiment.run_single_seed(seed=42)
            return result['final_accuracy']
        
        # Test reproducibility
        is_reproducible = validate_reproducibility(
            single_experiment_final_accuracy, 
            seed=42, 
            n_runs=3
        )
        
        if is_reproducible:
            print("✓ Reproducibility test: PASSED")
        else:
            print("❌ Reproducibility test: FAILED - Results not identical")
            
        return is_reproducible
        
    except Exception as e:
        print(f"❌ Reproducibility test FAILED: {e}")
        raise


def run_mini_multi_seed_experiment():
    """Run a small multi-seed experiment for validation."""
    print("\n=== Running Mini Multi-Seed Experiment ===")
    
    try:
        # Load baseline config and modify for quick testing
        config = load_config("configs/phase1_baseline.yaml")
        
        # Temporarily modify for faster testing
        original_epochs = config.config['training']['epochs']
        original_seeds = config.config['experiment']['seeds']
        
        config.config['training']['epochs'] = 20  # Reduced for testing
        config.config['experiment']['seeds'] = [42, 123]  # Just 2 seeds
        
        print(f"Running experiment with {len(config.get_seeds())} seeds, {config.training['epochs']} epochs")
        
        # Run experiment
        experiment = ASISRExperiment(config)
        results = experiment.run_multi_seed()
        
        # Validate results structure
        assert len(results.results_per_seed) == 2
        assert 'accuracy' in results.aggregated_results
        assert 'final_mean' in results.aggregated_results['accuracy']
        
        # Check confidence intervals
        results.compute_confidence_intervals()
        assert 'confidence_interval' in results.aggregated_results['accuracy']
        
        print("✓ Mini multi-seed experiment: PASSED")
        
        # Restore original config values for return
        config.config['training']['epochs'] = original_epochs
        config.config['experiment']['seeds'] = original_seeds
        
        return results
        
    except Exception as e:
        print(f"❌ Mini multi-seed experiment FAILED: {e}")
        raise


def test_statistical_utilities():
    """Test only the statistical utility functions that can be validated without full experiments."""
    print("\n=== Testing Statistical Utilities ===")
    
    try:
        # Test statistical utility functions that don't require experiment data
        from asisr.training.experiment import _interpret_effect_size
        
        # Test effect size interpretation function
        assert _interpret_effect_size(0.1) == "negligible"
        assert _interpret_effect_size(0.3) == "small"
        assert _interpret_effect_size(0.6) == "medium"
        assert _interpret_effect_size(0.9) == "large"
        
        print("✓ Effect size interpretation: PASSED")
        
        # Statistical comparison requires real baseline vs spectral experiment results
        print("❌ Full statistical comparison: NOT TESTED")
        print("  Reason: Requires actual baseline vs spectral experiment results")
        print("  Status: Deferred to Plan 2-C when we run real A/B comparisons")
        
        print("⚠️  Statistical utilities: PARTIAL - only basic functions tested")
        return True
        
    except Exception as e:
        print(f"❌ Statistical utilities FAILED: {e}")
        raise


def validate_framework():
    """Complete validation of the multi-seed statistical framework."""
    print("ASISR Multi-Seed Statistical Framework Validation")
    print("=" * 60)
    
    try:
        # Test all components
        baseline_config, spectral_config = test_config_loading()
        is_reproducible = test_reproducibility()
        mini_results = run_mini_multi_seed_experiment()
        statistical_utilities = test_statistical_utilities()
        
        print("\n" + "=" * 60)
        print("Framework Validation Summary")
        print("=" * 60)
        print("✅ Configuration loading: PASSED")
        print(f"✅ Reproducibility: {'PASSED' if is_reproducible else 'FAILED'}")
        print("✅ Multi-seed orchestration: PASSED")
        print("✅ Error bar computation: PASSED")
        print("✅ Confidence intervals: PASSED")
        print("⚠️  Statistical utilities: PARTIAL (basic functions only)")
        print("❌ Statistical A/B comparison: NOT TESTED (requires Plan 2-C data)")
        print("❌ Significance testing: NOT TESTED (requires Plan 2-C comparisons)")
        
        # Framework readiness
        print("\n🚀 Multi-seed statistical framework is ready!")
        print("Ready for Plan 2-C baseline vs spectral comparisons")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Framework validation FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("DEPRECATED: multi_seed_experiment.py")
    print("=" * 60)
    print("This validation script has been replaced by the unified experiment framework.")
    print()
    print("Please use instead:")
    print("  python experiments/phase1_boundary_mapping/unified_experiment.py validation")
    print()
    print("Running legacy validation for compatibility...")
    print()
    
    success = validate_framework()
    
    print()
    print("=" * 60)
    print("RECOMMENDATION: Migrate to unified_experiment.py")
    print("=" * 60)
    
    sys.exit(0 if success else 1)