#!/usr/bin/env python3
"""
Phase 4A: Focused Conditions Discovery

Strategic scale-up using proven TwoMoons/Circles infrastructure to test
core hypothesis: Do spectral regularization benefits emerge at realistic scales?
"""

import sys
from pathlib import Path
import time

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from spectra.experiments import Phase4AExperiment, Phase4AExperimentConfig


def run_focused_phase4a():
    """Run focused Phase 4A experiment on proven 2D datasets."""
    
    print("🎯 Phase 4A: Focused Conditions Discovery")
    print("="*60)
    print("Testing spectral regularization benefits at realistic scales")
    print("Focus: TwoMoons/Circles with architecture scaling + longer training")
    print()
    
    # Strategic experimental design based on proven infrastructure
    config = Phase4AExperimentConfig(
        # Architecture scaling: test capacity effects
        architectures=[
            ("8x8", [8, 8]),        # Phase 3 baseline (~120 params)
            ("16x16", [16, 16]),    # Phase 2D optimal (~464 params)  
            ("32x32", [32, 32]),    # Phase 2D peak (~1664 params)
            ("64x64", [64, 64]),    # Phase 2D diminishing (~5888 params)
            ("128x64", [128, 64])   # Asymmetric test (~12k params)
        ],
        
        # Proven datasets from Phase 2-3
        datasets=[
            "TwoMoons",  # Known to show +1.0% with linear scheduling 
            "Circles"    # Known complexity gradient from Phase 2C
        ],
        
        # Training regime scaling: test cumulative effects
        training_configs=[
            {"epochs": 100, "learning_rate": 0.01, "optimizer": "adam"},  # Phase 3 baseline
            {"epochs": 500, "learning_rate": 0.01, "optimizer": "adam"}   # 5x longer training
        ],
        
        # Spectral regularization methods: test best candidates
        regularization_configs=[
            {"type": "none"},  # Baseline
            {"type": "linear", "initial": 2.5, "final": 1.0, "strength": 0.1},  # Phase 2B validated
            {"type": "adaptive", "beta": -0.2, "strength": 0.1}  # Phase 3 implementation
        ],
        
        n_seeds=5,  # Statistical power
        device="auto"
    )
    
    total_experiments = (len(config.architectures) * len(config.datasets) * 
                        len(config.training_configs) * len(config.regularization_configs) * 
                        config.n_seeds)
    
    print(f"Experimental Design:")
    print(f"  Architectures: {len(config.architectures)} (8x8 → 128x64)")  
    print(f"  Datasets: {len(config.datasets)} (TwoMoons, Circles)")
    print(f"  Training configs: {len(config.training_configs)} (100, 500 epochs)")
    print(f"  Regularizations: {len(config.regularization_configs)} (none, linear, adaptive)")
    print(f"  Seeds per condition: {config.n_seeds}")
    print(f"  Total experiments: {total_experiments}")
    print()
    
    # Estimate execution time
    avg_time_per_exp = 2.0  # seconds, based on test results
    estimated_time = total_experiments * avg_time_per_exp / 60  # minutes
    print(f"Estimated execution time: {estimated_time:.1f} minutes")
    print()
    
    # Create and run experiment
    experiment = Phase4AExperiment(config)
    
    start_time = time.time()
    analysis = experiment.run()
    total_time = time.time() - start_time
    
    print(f"\n🎯 Phase 4A Focused Results Summary")
    print(f"="*50)
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"Successful experiments: {analysis['successful_experiments']}/{analysis['total_experiments']}")
    
    # Highlight key findings
    if analysis['recommendations']['decision'] == 'CONTINUE':
        print(f"🟢 RESULT: {analysis['recommendations']['decision']}")
        print(f"   Found promising conditions warranting full systematic sweep")
        print(f"   {analysis['summary_statistics']['n_meaningful_effects']} conditions with >1% improvement")
        
    elif analysis['recommendations']['decision'] == 'INVESTIGATE':
        print(f"🟡 RESULT: {analysis['recommendations']['decision']}")
        print(f"   Found 1 condition needing validation with more seeds")
        
    else:
        print(f"🔴 RESULT: {analysis['recommendations']['decision']}")
        print(f"   No meaningful benefits found in strategic scale-up")
        print(f"   Consider alternative research directions")
    
    print(f"\nRationale: {analysis['recommendations']['rationale']}")
    print(f"Next steps: {analysis['recommendations']['next_steps']}")
    
    return analysis


def update_project_docs(analysis):
    """Update project documentation with Phase 4A findings."""
    
    # Create Phase 4A results summary
    results_doc = Path("docs/phase4a-focused-results.md")
    
    with open(results_doc, 'w') as f:
        f.write("# Phase 4A: Focused Conditions Discovery Results\n\n")
        f.write(f"**Status**: {analysis['recommendations']['decision']}\n")
        f.write(f"**Execution Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Total Experiments**: {analysis['total_experiments']}\n")
        f.write(f"**Successful Runs**: {analysis['successful_experiments']}\n\n")
        
        f.write("## Experimental Design\n\n")
        f.write("**Strategic Focus**: Scale-up proven 2D infrastructure (TwoMoons/Circles)\n")
        f.write("**Hypothesis**: Spectral regularization benefits emerge at realistic scales\n\n")
        
        f.write("**Architectures Tested**:\n")
        f.write("- 8x8 (120 params) - Phase 3 baseline\n")
        f.write("- 16x16 (464 params) - Phase 2D optimal\n") 
        f.write("- 32x32 (1664 params) - Phase 2D peak\n")
        f.write("- 64x64 (5888 params) - Phase 2D diminishing\n")
        f.write("- 128x64 (12k params) - Asymmetric test\n\n")
        
        f.write("**Training Regimes**: 100 epochs (baseline), 500 epochs (cumulative effects)\n")
        f.write("**Regularization**: None (baseline), Linear (Phase 2B), Adaptive (Phase 3)\n\n")
        
        f.write("## Key Findings\n\n")
        if 'summary_statistics' in analysis and analysis['summary_statistics']:
            stats = analysis['summary_statistics']
            f.write(f"**Mean Relative Improvement**: {stats['mean_relative_improvement']:+.1%}\n")
            f.write(f"**Best Improvement Found**: {stats['max_improvement']:+.1%}\n")
            f.write(f"**Conditions with Positive Effects**: {stats['n_positive_effects']}/{len(analysis.get('comparisons', []))}\n")
            f.write(f"**Conditions with >1% Improvement**: {stats['n_meaningful_effects']}\n\n")
        
        f.write("## Decision Framework\n\n")
        f.write(f"**Recommendation**: {analysis['recommendations']['decision']}\n")
        f.write(f"**Rationale**: {analysis['recommendations']['rationale']}\n")
        f.write(f"**Next Steps**: {analysis['recommendations']['next_steps']}\n\n")
        
        if analysis.get('comparisons'):
            f.write("## Top Results\n\n")
            sorted_comparisons = sorted(
                analysis['comparisons'], 
                key=lambda x: x['improvement_relative'], 
                reverse=True
            )[:5]
            
            for i, comp in enumerate(sorted_comparisons, 1):
                f.write(f"{i}. **{comp['condition']}** ({comp['regularization_type']}): {comp['improvement_relative']:+.1%}\n")
                f.write(f"   - Baseline: {comp['baseline_accuracy']:.3f}\n")
                f.write(f"   - Spectral: {comp['spectral_accuracy']:.3f}\n")
                f.write(f"   - Effect size: {comp['effect_size']:.2f}\n\n")
    
    print(f"📄 Results documentation saved: {results_doc}")
    return results_doc


if __name__ == "__main__":
    print("🚀 Starting Phase 4A Focused Conditions Discovery")
    print("Testing spectral regularization at realistic scales...")
    print()
    
    try:
        analysis = run_focused_phase4a()
        
        # Update documentation
        results_doc = update_project_docs(analysis)
        
        print(f"\n✅ Phase 4A Focused Discovery Complete!")
        print(f"📊 Results saved to: plots/phase4a/")
        print(f"📄 Documentation: {results_doc}")
        
        # Suggest next steps
        if analysis['recommendations']['decision'] == 'CONTINUE':
            print(f"\n🎯 Suggested Next Action:")
            print(f"   Run full systematic sweep with MNIST integration")
            print(f"   Promising conditions warrant broader investigation")
            
        elif analysis['recommendations']['decision'] == 'INVESTIGATE':
            print(f"\n🔍 Suggested Next Action:")
            print(f"   Focus validation on the promising condition found")
            print(f"   Increase seeds and test robustness")
            
        else:
            print(f"\n🔄 Suggested Next Action:")
            print(f"   Consider pivoting research direction")
            print(f"   Archive systematic exploration results")
        
    except Exception as e:
        print(f"❌ Phase 4A experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)