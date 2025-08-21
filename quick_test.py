#!/usr/bin/env python3
"""Quick test to demonstrate the corrected analysis logic"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from spectra.experiments import Phase4AExperiment, Phase4AExperimentConfig


def main():
    """Quick test with fixed analysis logic."""
    
    print("🔧 Testing Fixed Analysis Logic")
    print("="*40)
    
    # Test both short (working) and medium (problematic) training
    configs = [
        ("short", 10),   # Known to show +2.3% improvement
        ("medium", 30)   # Test intermediate training length
    ]
    
    for name, epochs in configs:
        print(f"\n📊 Testing {name} training ({epochs} epochs)")
        print("-" * 40)
        
        config = Phase4AExperimentConfig(
            architectures=[("8x8", [8, 8])],
            datasets=["TwoMoons"],
            training_configs=[{"epochs": epochs, "learning_rate": 0.01, "optimizer": "adam"}],
            regularization_configs=[
                {"type": "none"},
                {"type": "linear_schedule", "initial_sigma": 2.5, "final_sigma": 1.0, "strength": 0.1}
            ],
            n_seeds=2,  # Quick test
            device="auto"
        )
        
        experiment = Phase4AExperiment(config)
        analysis = experiment.run()
        
        print(f"Results:")
        if analysis.get('comparisons'):
            for comp in analysis['comparisons']:
                print(f"  {comp['condition']}: {comp['improvement_relative']:+.1%}")
                print(f"    Baseline: {comp['baseline_accuracy']:.3f}")
                print(f"    Spectral: {comp['spectral_accuracy']:.3f}")
        
        print(f"Decision: {analysis['recommendations']['decision']}")
        print(f"Rationale: {analysis['recommendations']['rationale']}")


if __name__ == "__main__":
    main()