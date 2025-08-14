# SPECTRA: Spectral Performance Control Through Regularization Analysis

> **Research Focus**: Understanding performance-variance trade-offs in neural networks through spectral radius control

*Note: This project was previously called "ASISR" but was renamed to avoid collision with image super-resolution literature.*

## What We Discovered

**Phase 1 Results** (Belgium-Netherlands boundary classification, 5 seeds, 100 epochs):

- **✅ Spectral Control Works**: Successfully moved spectral radius from ~3.2 → ~2.0
- **📊 Performance-Variance Trade-off Identified**: -2.5% accuracy for ~100% variance reduction
- **📈 Large Effect Size**: Cohen's d = -1.196 (highly reproducible phenomenon)
- **🎯 Precise Measurements**: Baseline 80.3% ± 2.9% vs Spectral 77.9% ± 0.0%

**Key Insight**: Rather than improving performance, spectral regularization reveals a measurable trade-off between peak accuracy and training consistency.

## Quick Start

Run Phase 1 boundary learning experiment to reproduce findings:

```bash
# Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run unified experiment (comparison mode)
cd experiments/phase1_boundary_mapping
python unified_experiment.py --mode comparison --seeds 5 --epochs 100

# Results saved to: results/comparison_[timestamp]/
```

**Expected Output**: Statistical comparison showing spectral regularization's variance reduction with performance cost.

## Research Directions

**Current Investigation**: Characterizing the σ-performance-variance relationship across:
- **Spectral Operating Points**: Optimal σ selection for different applications
- **Application Domains**: Safety-critical systems, ensemble methods, production deployment  
- **Dynamic Control**: Training-phase-dependent spectral strategies
- **Multi-Dataset Validation**: Boundary complexity vs spectral requirements

**Next Phase**: Application-driven spectral control based on empirical trade-off characterization.

## Repository Structure

```
SPECTRA/
├── spectra/                     # Core research package
│   ├── models/                  # Spectral-regularized architectures
│   ├── regularization/          # Spectral control methods
│   ├── metrics/                 # Criticality assessment
│   └── training/                # Multi-seed experiment framework
├── experiments/                 # Phase-specific research
│   └── phase1_boundary_mapping/ # Belgium-Netherlands validation
├── configs/                     # YAML experiment configurations
└── docs/                        # Research documentation and papers
```

## Key Components

- **Unified Experiment Framework**: Single system supporting integration testing through full research experiments
- **Statistical Validation**: Multi-seed orchestration with proper significance testing
- **Spectral Control**: Precise spectral radius targeting via power iteration
- **Real Data**: Belgium-Netherlands boundary (Baarle-Nassau/Baarle-Hertog enclaves)

## Scientific Contributions

1. **Empirical Characterization**: First systematic measurement of spectral regularization's performance-variance trade-offs
2. **Reproducible Framework**: Statistical validation infrastructure for neural network criticality research
3. **Application Insights**: Identification of domains where consistency matters more than peak performance

## Data Attribution

Belgium-Netherlands boundary data from OpenStreetMap contributors, used under Open Database License.

---

**Status**: Phase 1 completed with statistical validation. Phase 2 planning based on empirical findings rather than theoretical assumptions.