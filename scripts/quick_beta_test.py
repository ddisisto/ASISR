#!/usr/bin/env python3
"""
Quick β parameter validation test.

Test different β values to confirm β=-0.2 is optimal for capacity scaling.
"""

import sys
from pathlib import Path
import yaml
import tempfile

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from spectra.utils.config import load_config
from spectra.training.experiment import SPECTRAExperiment


def test_beta_values():
    """Test different β values quickly."""
    print("🧪 QUICK β PARAMETER VALIDATION")
    print("=" * 35)
    
    beta_values = [-0.3, -0.2, -0.1, 0.0, 0.1]
    results = {}
    
    # Load base config
    base_config_path = Path("configs/phase3a_beta_test_quick.yaml")
    
    for beta in beta_values:
        print(f"\n🔬 Testing β = {beta}")
        
        # Load and modify config
        with open(base_config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config_data['regularization']['beta'] = beta
        config_data['experiment']['name'] = f"beta_test_{beta}"
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            # Run experiment
            config = load_config(Path(temp_config_path))
            experiment = SPECTRAExperiment(config)
            
            # Run with reduced seeds for speed
            result = experiment.run_multiple_seeds()
            final_accuracy = result.aggregated_results['accuracy']['final_mean']
            
            results[beta] = final_accuracy
            print(f"   Result: {final_accuracy:.3f} accuracy")
            
        except Exception as e:
            print(f"   Error: {e}")
            results[beta] = None
        finally:
            # Clean up temp file
            Path(temp_config_path).unlink()
    
    # Analyze results
    print(f"\n📊 β PARAMETER RESULTS:")
    best_beta = None
    best_accuracy = 0
    
    for beta, accuracy in results.items():
        if accuracy is not None:
            print(f"   β = {beta:5.1f}: {accuracy:.3f} accuracy")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_beta = beta
        else:
            print(f"   β = {beta:5.1f}: FAILED")
    
    print(f"\n🎯 OPTIMAL β PARAMETER:")
    if best_beta is not None:
        print(f"   Best: β = {best_beta} ({best_accuracy:.3f} accuracy)")
        if best_beta == -0.2:
            print("   ✅ CONFIRMED: β=-0.2 is optimal!")
        else:
            print(f"   📝 NOTE: β={best_beta} slightly better than β=-0.2")
    
    return results


if __name__ == "__main__":
    test_beta_values()