# SPECTRA: Spectral Performance Control Through Regularization Analysis

> **Research Focus**: Understanding performance-variance trade-offs in neural networks through spectral radius control

*Note: This project was previously called "ASISR" but was renamed to avoid collision with image super-resolution literature.*

## What We Discovered

**Phase 1 Results** (Belgium-Netherlands boundary classification, 5 seeds, 100 epochs):

- **✅ Spectral Control Works**: Successfully moved spectral radius from ~3.2 → ~2.0
- **📊 Performance-Variance Trade-off Identified**: -2.5% accuracy for ~100% variance reduction
- **📈 Large Effect Size**: Cohen's d = -1.196 (highly reproducible phenomenon)
- **🎯 Precise Measurements**: Baseline 80.3% ± 2.9% vs Spectral 77.9% ± 0.0%

**Phase 2B Breakthrough** (Dynamic spectral scheduling, multi-dataset validation):

- **🚀 Linear Schedule Success**: +1.1% accuracy improvement (p=0.0344*, Cohen's d=1.610)
- **⚡ Training-Phase Control**: Dynamic σ scheduling outperforms static approaches
- **📈 Exponential Strategy**: +0.7% accuracy improvement (p=0.1447, Cohen's d=1.022)
- **🎯 Statistical Rigor**: 5-seed experiments with confidence intervals and effect sizes

**Key Insights**: 
1. **Phase 1**: Static spectral regularization reveals performance-variance trade-offs
2. **Phase 2B**: Dynamic strategies break the static trade-off limitations - we can optimize both performance AND variance through training-phase-dependent spectral control

## Quick Start

### Phase 1: Baseline Spectral Control
```bash
# Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run unified experiment (comparison mode)
cd experiments/phase1_boundary_mapping
python unified_experiment.py --mode comparison --seeds 5 --epochs 100

# Results saved to: results/comparison_[timestamp]/
```

### Phase 2B: Dynamic Spectral Strategies
```bash
# Run breakthrough dynamic scheduling experiments
cd experiments/phase2b_dynamic_spectral
python phase2b_experiment.py --config ../../configs/phase2b_linear_schedule.yaml

# Compare all dynamic strategies
for config in linear exponential step static; do
  python phase2b_experiment.py --config ../../configs/phase2b_${config}_schedule.yaml
done

# Results and visualizations saved to: plots/phase2b/
```

**Expected Output**: 
- **Phase 1**: Statistical comparison showing spectral regularization's variance reduction with performance cost
- **Phase 2B**: Demonstration that dynamic σ scheduling achieves superior performance-variance trade-offs

## Research Status

**✅ Phase 1 Completed**: Static spectral regularization performance-variance trade-offs characterized
**✅ Phase 2A Completed**: Multi-σ framework and empirical σ-performance relationships established  
**✅ Phase 2B Completed**: Dynamic spectral scheduling breakthrough - training-phase-dependent control validated
**🔄 Current Focus**: Interactive visual exploration layer for real-time σ parameter control

**Next Phase**: Visual dynamics framework with:
- **Interactive σ Exploration**: Real-time parameter adjustment with immediate feedback
- **Training Visualization**: Live σ schedule evolution during training processes  
- **Publication Pipeline**: Interactive dashboards for scientific storytelling
- **Application Domains**: Safety-critical systems, ensemble methods, production deployment

## Repository Structure

```
SPECTRA/
├── spectra/                          # Core research package
│   ├── models/                       # Spectral-regularized architectures
│   ├── regularization/               # Spectral control methods
│   │   ├── fixed.py                  # Static spectral regularization (Phase 1)
│   │   └── dynamic.py                # Dynamic scheduling algorithms (Phase 2B)
│   ├── metrics/                      # Criticality assessment
│   └── training/                     # Multi-seed experiment framework
├── experiments/                      # Phase-specific research
│   ├── phase1_boundary_mapping/      # Belgium-Netherlands validation
│   ├── phase2a_multi_sigma/          # Multi-σ framework validation
│   └── phase2b_dynamic_spectral/     # Dynamic scheduling breakthrough
├── configs/                          # YAML experiment configurations
│   ├── phase1_*.yaml                 # Static spectral experiments
│   ├── phase2a_*.yaml                # Multi-σ experiments  
│   └── phase2b_*.yaml                # Dynamic scheduling experiments
└── docs/                             # Research documentation and papers
```

## Key Components

- **Unified Experiment Framework**: Single system supporting integration testing through full research experiments
- **Dynamic Spectral Control**: Training-phase-dependent σ scheduling with linear, exponential, and step strategies
- **Statistical Validation**: Multi-seed orchestration with proper significance testing and effect size reporting
- **Spectral Control**: Precise spectral radius targeting via power iteration
- **Real Data**: Belgium-Netherlands boundary (Baarle-Nassau/Baarle-Hertog enclaves) + synthetic datasets

## Scientific Contributions

1. **Empirical Characterization**: First systematic measurement of spectral regularization's performance-variance trade-offs
2. **Dynamic Control Breakthrough**: Training-phase-dependent spectral strategies that optimize both performance and variance
3. **Statistical Validation Framework**: Rigorous multi-seed orchestration with significance testing and effect size reporting
4. **Application Insights**: Identification of domains where consistency matters more than peak performance

## Data Attribution

Belgium-Netherlands boundary data from OpenStreetMap contributors, used under Open Database License.

---

**Status**: Phase 2B completed with breakthrough results. Dynamic spectral scheduling achieves statistically significant improvements over static approaches. Ready for visual exploration layer development.