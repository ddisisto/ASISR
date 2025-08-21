#!/usr/bin/env python3
"""
Phase 4A: Pilot Experiment 

Quick pilot to validate approach before full focused discovery.
Tests key hypothesis on minimal but representative subset.
"""

import sys
from pathlib import Path
import time

sys.path.append(str(Path(__file__).parent))

from spectra.experiments import Phase4AExperiment, Phase4AExperimentConfig


def main():
    """Run pilot Phase 4A experiment."""
    
    print("🧪 Phase 4A: Pilot Experiment")
    print("="*40)
    print("Quick validation before full focused discovery")
    print()
    
    # Minimal but representative test
    config = Phase4AExperimentConfig(
        architectures=[
            ("8x8", [8, 8]),        # Baseline
            ("32x32", [32, 32])     # Scale-up test
        ],
        datasets=["TwoMoons"],      # Known working dataset
        training_configs=[
            {"epochs": 50, "learning_rate": 0.01, "optimizer": "adam"}  # Quick training
        ],
        regularization_configs=[
            {"type": "none"},       # Baseline
            {"type": "linear", "initial": 2.5, "final": 1.0, "strength": 0.1}  # Phase 2B method
        ],
        n_seeds=3,  # Minimal for speed
        device="auto"
    )
    
    total_experiments = (len(config.architectures) * len(config.datasets) * 
                        len(config.training_configs) * len(config.regularization_configs) * 
                        config.n_seeds)
    
    print(f"Pilot Design: {total_experiments} experiments")
    print(f"Testing: spectral vs baseline on 8x8 & 32x32 architectures")
    print()
    
    start_time = time.time()
    experiment = Phase4AExperiment(config)
    analysis = experiment.run()
    total_time = time.time() - start_time
    
    print(f"\n🎯 Pilot Results")
    print(f"="*30)
    print(f"Execution time: {total_time:.1f} seconds")
    print(f"Successful: {analysis['successful_experiments']}/{analysis['total_experiments']}")
    
    if analysis.get('comparisons'):
        print(f"\nKey Comparisons:")
        for comp in analysis['comparisons']:
            print(f"  {comp['condition']} ({comp['regularization_type']}): {comp['improvement_relative']:+.1%}")
    
    print(f"\nRecommendation: {analysis['recommendations']['decision']}")
    print(f"Rationale: {analysis['recommendations']['rationale']}")
    
    # Save pilot results
    results_file = Path("plots/phase4a/pilot_results.json")
    results_file.parent.mkdir(exist_ok=True)
    
    import json
    with open(results_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n💾 Pilot results saved: {results_file}")
    
    if analysis['recommendations']['decision'] in ['CONTINUE', 'INVESTIGATE']:
        print(f"✅ Pilot successful - proceed with full focused discovery")
    else:
        print(f"⚠️  Pilot shows no benefits - reconsider approach")
    
    return analysis


if __name__ == "__main__":
    main()