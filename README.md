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

### Unified CLI Interface
```bash
# Setup environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# List available experiments
python run_experiment.py list

# Run Phase 2B comprehensive comparison
python run_experiment.py phase2b \
  --static configs/phase2b_static_comparison.yaml \
  --dynamic configs/phase2b_linear_schedule.yaml \
           configs/phase2b_exponential_schedule.yaml \
           configs/phase2b_step_schedule.yaml \
  --names Linear Exponential Step \
  --plots

# Run single experiment
python run_experiment.py single configs/phase2b_linear_schedule.yaml

# Results saved to standardized paths: plots/phase*/
```

### Legacy Phase-Specific Commands
```bash
# Phase 1: Boundary mapping (legacy interface)
cd experiments/phase1_boundary_mapping
python unified_experiment.py --mode comparison --seeds 5 --epochs 100
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
│   ├── experiments/                  # Modular experiment framework
│   │   ├── base.py                   # Abstract interfaces and result containers
│   │   └── phase2b.py                # Phase 2B dynamic vs static experiments
│   ├── metrics/                      # Criticality assessment
│   ├── training/                     # Multi-seed experiment framework with trajectory collection
│   └── visualization/                # Analysis and plotting tools
│       └── schedules.py              # σ scheduling visualization suite
├── experiments/                      # Phase-specific research (legacy)
│   ├── phase1_boundary_mapping/      # Belgium-Netherlands validation
│   ├── phase2a_multi_sigma/          # Multi-σ framework validation
│   └── phase2b_dynamic_spectral/     # Dynamic scheduling breakthrough
├── plots/                            # Standardized output structure
│   ├── phase1/                       # Phase 1 results
│   ├── phase2b/                      # Phase 2B dynamic scheduling results
│   └── phase2c/                      # Phase 2C visualization results
├── configs/                          # YAML experiment configurations
├── run_experiment.py                 # Unified CLI for all experiments
└── docs/                             # Research documentation and papers
```

## Key Components

- **Modular Experiment Framework**: Clean separation between core training, experiment orchestration, and visualization
- **Unified CLI Interface**: `run_experiment.py` provides consistent access to all experimental phases
- **Complete Trajectory Collection**: Every-epoch criticality metrics enable detailed training dynamics analysis
- **Dynamic Spectral Control**: Training-phase-dependent σ scheduling with linear, exponential, and step strategies
- **Statistical Validation**: Multi-seed orchestration with proper significance testing and effect size reporting
- **Standardized Outputs**: All results saved to `plots/phase*/` with consistent naming and structure
- **Publication-Quality Visualization**: Professional plots ready for scientific papers

## Scientific Contributions

1. **Empirical Characterization**: First systematic measurement of spectral regularization's performance-variance trade-offs
2. **Dynamic Control Breakthrough**: Training-phase-dependent spectral strategies that optimize both performance and variance
3. **Statistical Validation Framework**: Rigorous multi-seed orchestration with significance testing and effect size reporting
4. **Application Insights**: Identification of domains where consistency matters more than peak performance

## Data Attribution

Belgium-Netherlands boundary data from OpenStreetMap contributors, used under Open Database License.

---

**Status**: Phase 2B completed with breakthrough results. Dynamic spectral scheduling achieves statistically significant improvements over static approaches. Ready for visual exploration layer development.