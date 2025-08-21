#!/usr/bin/env python3
"""
Phase 4A Infrastructure Test Script

Quick test to validate that Phase 4A experiment infrastructure is working
before running full experimental suite.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from spectra.experiments import Phase4AExperiment, Phase4AExperimentConfig
from spectra.utils.seed import set_seed


def test_minimal_experiment():
    """Test minimal Phase 4A experiment to validate infrastructure."""
    print("🧪 Testing Phase 4A Infrastructure")
    print("="*50)
    
    # Create minimal test configuration
    config = Phase4AExperimentConfig(
        # Minimal strategic sampling for testing
        architectures=[("8x8", [8, 8])],  # Just one architecture
        datasets=["TwoMoons"],             # Just one dataset
        training_configs=[{"epochs": 10, "learning_rate": 0.01, "optimizer": "adam"}],  # Short training
        regularization_configs=[{"type": "none"}],  # Just baseline
        n_seeds=2,  # Minimal seeds
        device="auto"
    )
    
    print(f"Test config:")
    print(f"  Architectures: {len(config.architectures)}")
    print(f"  Datasets: {len(config.datasets)}")
    print(f"  Training configs: {len(config.training_configs)}")
    print(f"  Regularizations: {len(config.regularization_configs)}")
    print(f"  Seeds: {config.n_seeds}")
    print(f"  Total experiments: {len(config.architectures) * len(config.datasets) * len(config.training_configs) * len(config.regularization_configs) * config.n_seeds}")
    
    # Create experiment
    experiment = Phase4AExperiment(config)
    
    # Run just baseline experiments
    print(f"\n🎯 Running baseline experiments...")
    try:
        baseline_results = experiment.run_baseline_experiments()
        
        print(f"✅ Baseline experiments completed!")
        print(f"  Results: {len(baseline_results)} conditions")
        
        if baseline_results and "error" not in baseline_results[0]:
            result = baseline_results[0]
            print(f"  Sample result: {result['accuracy_mean']:.3f} ± {result['accuracy_std']:.3f} accuracy")
            print(f"  Total time: {result['total_execution_time']:.1f} seconds")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mnist_loading():
    """Test MNIST loading capability."""
    print("\n📊 Testing MNIST Loading")
    print("="*30)
    
    try:
        from spectra.data import create_real_dataset_loader
        
        # Test MNIST loader
        mnist_loader = create_real_dataset_loader("MNIST", train=True, flatten=True, normalize=True)
        features, labels = mnist_loader.load_data()
        
        print(f"✅ MNIST loaded successfully!")
        print(f"  Features shape: {features.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Features range: [{features.min():.3f}, {features.max():.3f}]")
        print(f"  Unique labels: {torch.unique(labels).tolist()}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  MNIST loading failed: {e}")
        print("  This is OK if torchvision is not available")
        return False


if __name__ == "__main__":
    import torch
    
    print("🚀 Phase 4A Infrastructure Validation")
    print("="*60)
    
    # Test device availability
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU: Not available, using CPU")
    
    # Run tests
    success = True
    
    success &= test_minimal_experiment()
    test_mnist_loading()  # Optional test
    
    print(f"\n{'='*60}")
    if success:
        print("✅ Phase 4A Infrastructure Ready!")
        print("Ready to proceed with full experimental suite.")
    else:
        print("❌ Infrastructure issues detected.")
        print("Please resolve errors before proceeding.")
    print("="*60)