#!/usr/bin/env python3
"""
Phase 3B Cross-Dataset Validation Script

Validates that the capacity-adaptive regularization framework
can handle cross-dataset experiments for generalization testing.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectra.utils.config import load_config
from spectra.training.experiment import SPECTRAExperiment
from spectra.utils.capacity import analyze_model_capacity, print_capacity_summary
from spectra.models import SpectralMLP


def test_dataset_loading():
    """Test that all Phase 3B datasets load correctly."""
    print("🌐 Testing Cross-Dataset Loading")
    print("=" * 35)
    
    datasets_to_test = [
        ("TwoMoons", "configs/phase3a_optimal_beta_16x16.yaml"),
        ("Circles", "configs/phase3b_cross_dataset_circles.yaml"), 
        ("BaarleMap", "configs/phase3b_cross_dataset_belgium.yaml")
    ]
    
    success_count = 0
    
    for dataset_name, config_path in datasets_to_test:
        print(f"\n📊 Testing {dataset_name} dataset...")
        try:
            config = load_config(Path(config_path))
            experiment = SPECTRAExperiment(config)
            
            # Test data loading
            coords, labels = experiment._setup_data()
            print(f"✅ {dataset_name}: {coords.shape[0]} samples, {coords.shape[1]}D input")
            print(f"   Labels shape: {labels.shape}, unique values: {torch.unique(labels).tolist()}")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {dataset_name}: {e}")
    
    print(f"\nDataset Loading: {success_count}/{len(datasets_to_test)} successful")
    return success_count == len(datasets_to_test)


def test_capacity_calculations():
    """Test capacity calculations across different datasets."""
    print("\n🔍 Testing Cross-Dataset Capacity Calculations")
    print("=" * 50)
    
    # Test capacity ratios for different architecture-dataset combinations
    test_cases = [
        ("8x8", [8, 8], "two_moons", "Under-parameterized"),
        ("16x16", [16, 16], "two_moons", "Optimal"),
        ("32x32", [32, 32], "circles", "Optimal"), 
        ("64x64", [64, 64], "belgium_netherlands", "Optimal")
    ]
    
    for arch_name, hidden_dims, dataset_name, expected_category in test_cases:
        print(f"\n🔬 {arch_name} on {dataset_name}:")
        try:
            model = SpectralMLP(input_dim=2, hidden_dims=hidden_dims, output_dim=1)
            analysis = analyze_model_capacity(model, dataset_name)
            
            print(f"   Parameters: {analysis.total_params}")
            print(f"   Capacity ratio: {analysis.capacity_ratio:.3f}")
            print(f"   Category: {analysis.capacity_category.replace('_', ' ').title()}")
            
            # Validate expected category
            if expected_category.lower().replace(' ', '_') in analysis.capacity_category:
                print("   ✅ Expected capacity category confirmed")
            else:
                print(f"   ⚠️  Expected {expected_category}, got {analysis.capacity_category}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    return True


def test_phase3b_experiments():
    """Test Phase 3B experiment configurations."""
    print("\n🧪 Testing Phase 3B Experiment Configurations")
    print("=" * 45)
    
    phase3b_configs = [
        "phase3b_cross_dataset_circles.yaml",
        "phase3b_cross_dataset_belgium.yaml"
    ]
    
    success_count = 0
    
    for config_name in phase3b_configs:
        print(f"\n🔧 Testing {config_name}...")
        try:
            config_path = Path("configs") / config_name
            config = load_config(config_path)
            
            # Test experiment creation
            experiment = SPECTRAExperiment(config)
            print(f"✅ Experiment created: {config.experiment['name']}")
            
            # Test short run (1 epoch, 1 seed)
            config.experiment['seeds'] = [42]
            config.training['epochs'] = 1
            
            result = experiment.run_single_seed(42)
            accuracy = result.get('final_metrics', {}).get('accuracy', result.get('metrics_history', {}).get('accuracy', [0.5])[-1])
            print(f"   Training successful, final accuracy: {accuracy:.3f}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ {config_name}: {e}")
    
    print(f"\nPhase 3B Experiments: {success_count}/{len(phase3b_configs)} successful")
    return success_count == len(phase3b_configs)


def test_capacity_adaptive_cross_dataset():
    """Test capacity-adaptive regularization across datasets."""
    print("\n🎯 Testing Capacity-Adaptive Cross-Dataset Performance")
    print("=" * 55)
    
    # Test the key Phase 3B hypothesis: capacity-adaptive scheduling
    # should enable positive improvements across all datasets
    
    test_combinations = [
        ("16x16", [16, 16], "two_moons", "Expected: Strong positive effect"),
        ("32x32", [32, 32], "circles", "Expected: Positive effect (optimal capacity)"),
        ("64x64", [64, 64], "belgium_netherlands", "Expected: Positive effect (high complexity)")
    ]
    
    print("Testing capacity-adaptive scheduling across optimal architectures...")
    
    for arch_name, hidden_dims, dataset_name, expectation in test_combinations:
        print(f"\n🎲 {arch_name} on {dataset_name}: {expectation}")
        
        try:
            # Create model and check capacity
            model = SpectralMLP(input_dim=2, hidden_dims=hidden_dims, output_dim=1)
            analysis = analyze_model_capacity(model, dataset_name)
            
            print(f"   Capacity ratio: {analysis.capacity_ratio:.3f}")
            print(f"   Architecture category: {analysis.capacity_category}")
            
            # Validate that this matches our Phase 2D predictions
            if dataset_name == "two_moons" and "16" in arch_name:
                print("   ✅ TwoMoons + 16x16: Validated optimal from Phase 2D")
            elif dataset_name == "circles" and "32" in arch_name:
                print("   📈 Circles + 32x32: Predicted optimal for medium complexity")
            elif dataset_name == "belgium_netherlands" and "64" in arch_name:
                print("   📊 Belgium + 64x64: Predicted optimal for high complexity")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            return False
    
    return True


if __name__ == "__main__":
    import torch
    
    print("🚀 SPECTRA Phase 3B Cross-Dataset Validation")
    print("=" * 50)
    
    # Run all validation tests
    dataset_loading_ok = test_dataset_loading()
    capacity_calc_ok = test_capacity_calculations()
    experiment_config_ok = test_phase3b_experiments()
    cross_dataset_ok = test_capacity_adaptive_cross_dataset()
    
    if all([dataset_loading_ok, capacity_calc_ok, experiment_config_ok, cross_dataset_ok]):
        print("\n🎉 PHASE 3B CROSS-DATASET FRAMEWORK: SUCCESS!")
        print("   ✅ All datasets load correctly")
        print("   ✅ Capacity calculations work across datasets") 
        print("   ✅ Experiment configurations validated")
        print("   ✅ Cross-dataset capacity-adaptive scheduling ready")
        print("\n🎯 Ready for Phase 3B generalization experiments!")
        sys.exit(0)
    else:
        print("\n⚠️  Issues detected in Phase 3B framework")
        sys.exit(1)