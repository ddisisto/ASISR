#!/usr/bin/env python3
"""
Phase 3A Framework Validation Script

Quick validation that the capacity-adaptive regularization framework
is correctly integrated and can run Phase 3A experiments.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectra.utils.config import load_config
from spectra.training.experiment import SPECTRAExperiment
from spectra.utils.capacity import analyze_model_capacity, print_capacity_summary
from spectra.models import SpectralMLP


def test_capacity_adaptive_regularizer():
    """Test capacity-adaptive regularizer creation and basic functionality."""
    print("🔬 Testing Capacity-Adaptive Regularizer Framework")
    print("=" * 60)
    
    # Test 1: Load Phase 3A configuration
    print("\n1. Loading Phase 3A configuration...")
    try:
        config_path = Path("configs/phase3a_optimal_beta_8x8.yaml") 
        config = load_config(config_path)
        print(f"✅ Successfully loaded config: {config.experiment['name']}")
        print(f"   Description: {config.experiment['description']}")
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False
    
    # Test 2: Create experiment instance
    print("\n2. Creating SPECTRA experiment...")
    try:
        experiment = SPECTRAExperiment(config)
        print("✅ Successfully created SPECTRAExperiment instance")
    except Exception as e:
        print(f"❌ Failed to create experiment: {e}")
        return False
    
    # Test 3: Create model and analyze capacity
    print("\n3. Testing capacity analysis...")
    try:
        model = SpectralMLP(input_dim=2, hidden_dims=[8, 8], output_dim=1)  # 8x8 architecture for testing
        analysis = analyze_model_capacity(model, 'two_moons')
        print("✅ Successfully analyzed model capacity:")
        print_capacity_summary(analysis)
    except Exception as e:
        print(f"❌ Failed capacity analysis: {e}")
        return False
    
    # Test 4: Test regularizer creation (single seed)
    print("\n4. Testing capacity-adaptive regularizer creation...")
    try:
        # Create a minimal test by modifying the config directly for short test
        original_seeds = config.experiment['seeds']
        original_epochs = config.training['epochs']
        
        config.experiment['seeds'] = [42]  # Single seed
        config.training['epochs'] = 2      # Very short for testing
        
        test_experiment = SPECTRAExperiment(config)
        
        # Restore original values
        config.experiment['seeds'] = original_seeds
        config.training['epochs'] = original_epochs
        result = test_experiment.run_single_seed(42)
        
        print("✅ Successfully created and ran capacity-adaptive regularizer")
        print(f"   Final accuracy: {result['final_metrics']['accuracy']:.4f}")
        if 'capacity_ratio' in result['final_metrics']:
            print(f"   Capacity ratio: {result['final_metrics']['capacity_ratio']:.3f}")
    except Exception as e:
        print(f"❌ Failed regularizer test: {e}")
        return False
    
    print("\n🎉 Phase 3A Framework Validation: SUCCESS!")
    print("   Ready for β parameter sweep experiments")
    return True


def test_configuration_matrix():
    """Test that all Phase 3 configurations can be loaded."""
    print("\n🔧 Testing Phase 3 Configuration Matrix")
    print("=" * 40)
    
    phase3_configs = [
        "phase3a_beta_sweep_twomoons.yaml",
        "phase3a_optimal_beta_8x8.yaml", 
        "phase3a_optimal_beta_16x16.yaml",
        "phase3b_cross_dataset_belgium.yaml",
        "phase3b_cross_dataset_circles.yaml",
        "phase3c_universal_validation.yaml"
    ]
    
    configs_path = Path("configs")
    success_count = 0
    
    for config_name in phase3_configs:
        config_path = configs_path / config_name
        try:
            config = load_config(config_path)
            print(f"✅ {config_name}: {config.experiment['name']}")
            success_count += 1
        except Exception as e:
            print(f"❌ {config_name}: {e}")
    
    print(f"\nConfiguration Matrix: {success_count}/{len(phase3_configs)} successful")
    return success_count == len(phase3_configs)


if __name__ == "__main__":
    print("🚀 SPECTRA Phase 3A Framework Validation")
    print("=" * 50)
    
    # Run framework validation
    framework_ok = test_capacity_adaptive_regularizer()
    
    # Run configuration validation  
    configs_ok = test_configuration_matrix()
    
    if framework_ok and configs_ok:
        print("\n🎯 PHASE 3A READY FOR EXECUTION!")
        print("   All systems validated - ready to begin β parameter optimization")
        sys.exit(0)
    else:
        print("\n⚠️  Issues detected - see errors above")
        sys.exit(1)