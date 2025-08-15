# Phase 1 Plan 2-B: Multi-Seed Statistical Framework

**Sub-phase Focus**: Implement scientific rigor with 5+ seed experimental validation and proper statistical analysis.

**Context Budget**: ~40KB (framework + configuration system)

**Dependencies**: Plan 2-A criticality implementation complete

## Objective

Build the statistical validation framework required for scientific claims about spectral regularization benefits. This includes multi-seed experiment orchestration, configuration management, and statistical analysis tools.

## Implementation Scope

### **Primary Targets**
- `spectra/training/experiment.py` - Formal multi-seed experiment runner
- `configs/phase1_baseline.yaml` - Baseline experiment configuration  
- `configs/phase1_spectral.yaml` - Spectral regularization configuration
- `experiments/phase1_boundary_mapping/multi_seed_experiment.py` - Statistical validation script

### **Secondary Targets**
- `spectra/utils/seed.py` - Deterministic seed management
- `spectra/utils/config.py` - YAML configuration loading and validation
- Statistical analysis utilities for aggregating multi-seed results

### **Strict Scope Boundaries**
- ❌ **No A/B comparison experiments** (Plan 2-C)
- ❌ **No publication visualization** (Plan 2-D)
- ❌ **No adaptive regularization** (Phase 2)

## Success Criteria

### **Minimum Success**
- Multi-seed experiments run reproducibly (5+ seeds)
- Configuration-driven experiment setup working
- Statistical aggregation of results across seeds
- Proper error bar computation

### **Target Success**
- Statistical significance testing framework operational
- Effect size reporting with confidence intervals
- Reproducibility validation across different environments
- Professional statistical analysis ready for Plan 2-C comparisons

## Implementation Strategy

### **Step 1**: Configuration Management *(Priority 1)*
```yaml
# configs/phase1_baseline.yaml
model:
  type: "SpectralMLP"
  hidden_dims: [64, 64]
  activation: "relu"

data:
  type: "BaarleMap" 
  resolution: 200
  bounds: [-1.5, 2.5, -1.0, 1.5]

training:
  epochs: 100
  learning_rate: 1e-3
  batch_size: 1000

regularization:
  type: null  # Baseline comparison

experiment:
  seeds: [42, 123, 456, 789, 1011]
  metrics: ["accuracy", "loss", "criticality_score"]
```

### **Step 2**: Multi-Seed Orchestration *(Priority 2)*
```python
# spectra/training/experiment.py
class SPECTRAExperiment:
    def run_multi_seed(self, config, n_seeds=5):
        """Run experiment across multiple seeds with statistical aggregation"""
        # Seed management
        # Result collection
        # Statistical analysis
        # Error bar computation
```

### **Step 3**: Statistical Analysis *(Priority 3)*
- Results aggregation across seeds
- Statistical significance testing
- Effect size computation
- Confidence interval calculation

## Engineering Patterns Application

**Reference**: [docs/engineering-patterns.md](./docs/engineering-patterns.md)

- **Configuration-driven experiments**: YAML files for reproducibility
- **5+ seeds minimum**: Statistical rigor for all claims
- **Working main always**: Frequent commits with functional code
- **Real data throughout**: No synthetic datasets on critical path

## Branch Workflow

```bash
git checkout -b plan2b-multiseed
# Implement statistical framework
# Test configuration system
# Validate multi-seed reproducibility
git checkout main && git merge plan2b-multiseed
git branch -d plan2b-multiseed
```

## Handover Preparation

**Completion triggers**:
- Multi-seed experiments run successfully
- Statistical analysis framework operational
- Configuration management working
- Reproducibility validated

**Next context setup**:
- Update this document with COMPLETED status
- Create `docs/phase1-plan2c-comparison.md` for A/B testing
- Commit with statistical framework ready for baseline vs spectral comparisons

## Quality Gates

- [ ] YAML configuration system working
- [ ] Multi-seed experiment orchestration functional
- [ ] Statistical significance testing implemented
- [ ] Effect size reporting with confidence intervals
- [ ] Reproducibility validated across seeds
- [ ] No scope creep beyond statistical framework

**Context Budget Target**: Complete within single focused session (~40KB)