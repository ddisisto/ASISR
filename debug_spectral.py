#!/usr/bin/env python3
"""Debug spectral regularization issue in Phase 4A"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from spectra.experiments import Phase4AExperiment, Phase4AExperimentConfig


def test_single_spectral_experiment():
    """Test a single spectral regularization experiment to isolate the issue."""
    
    print("🔬 Testing Single Spectral Regularization Experiment")
    print("="*60)
    
    config = Phase4AExperimentConfig(
        architectures=[("8x8", [8, 8])],
        datasets=["TwoMoons"],
        training_configs=[{"epochs": 10, "learning_rate": 0.01, "optimizer": "adam"}],
        regularization_configs=[{"type": "linear_schedule", "initial_sigma": 2.5, "final_sigma": 1.0, "strength": 0.1}],
        n_seeds=1,
        device="auto"
    )
    
    experiment = Phase4AExperiment(config)
    
    print("Testing single condition...")
    try:
        result = experiment.run_single_condition(
            "8x8", [8, 8], "TwoMoons",
            {"epochs": 10, "learning_rate": 0.01, "optimizer": "adam"},
            {"type": "linear_schedule", "initial_sigma": 2.5, "final_sigma": 1.0, "strength": 0.1}
        )
        
        print(f"✅ Success! Result: {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f}")
        return True
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_baseline_vs_spectral():
    """Test the comparison logic between baseline and spectral."""
    
    print("\n🔬 Testing Baseline vs Spectral Comparison")
    print("="*60)
    
    config = Phase4AExperimentConfig(
        architectures=[("8x8", [8, 8])],
        datasets=["TwoMoons"],
        training_configs=[{"epochs": 10, "learning_rate": 0.01, "optimizer": "adam"}],
        regularization_configs=[
            {"type": "none"},
            {"type": "linear_schedule", "initial_sigma": 2.5, "final_sigma": 1.0, "strength": 0.1}
        ],
        n_seeds=1,
        device="auto"
    )
    
    experiment = Phase4AExperiment(config)
    
    try:
        # Test baseline
        print("Running baseline...")
        baseline_results = experiment.run_baseline_experiments()
        print(f"✅ Baseline: {len(baseline_results)} results")
        
        # Test spectral  
        print("Running spectral...")
        spectral_results = experiment.run_spectral_experiments(baseline_results)
        print(f"✅ Spectral: {len(spectral_results)} results")
        
        # Test analysis
        print("Running analysis...")
        analysis = experiment.analyze_results(baseline_results, spectral_results)
        print(f"✅ Analysis: {len(analysis.get('comparisons', []))} comparisons")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed at comparison: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("🧪 Debugging Phase 4A Spectral Regularization")
    print()
    
    success1 = test_single_spectral_experiment()
    success2 = test_baseline_vs_spectral()
    
    if success1 and success2:
        print("\n✅ All tests passed - issue might be in full experiment flow")
    else:
        print("\n❌ Found specific failure points to fix")