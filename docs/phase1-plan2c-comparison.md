# Phase 1 Plan 2-C: Baseline vs Spectral A/B Testing

**Sub-phase Focus**: Execute controlled experiments comparing baseline and spectral regularization to validate core SPECTRA hypothesis.

**Context Budget**: ~50KB (experiments + analysis)

**Dependencies**: Plans 2-A (criticality) and 2-B (multi-seed framework) complete

## Objective

Answer the core Phase 1 research question: *"What effects does spectral regularization at σ ≈ 1.0 have on neural network learning dynamics?"*

Execute rigorous A/B testing with statistical validation to demonstrate spectral regularization benefits.

## Implementation Scope

### **Primary Targets**
- `experiments/phase1_boundary_mapping/baseline_vs_spectral.py` - Head-to-head comparison experiments
- Statistical analysis of training efficiency differences
- Decision boundary quality assessment
- Criticality indicator analysis during training

### **Secondary Targets**
- Training dynamics monitoring and logging
- Convergence analysis (epochs to target accuracy)
- Spectral radius evolution tracking
- Effect size quantification

### **Strict Scope Boundaries**
- ❌ **No publication visualization** (Plan 2-D)
- ❌ **No adaptive regularization** (Phase 2)
- ❌ **No architecture exploration** (Phase 3)

## Success Criteria

### **Minimum Success**
- Controlled A/B experiments run successfully
- Statistical comparison of baseline vs spectral performance
- Clear quantification of training efficiency differences
- Criticality indicator correlation analysis

### **Target Success**
- **10-20% training efficiency improvement** with spectral regularization
- **Statistical significance** (p < 0.05) with substantial effect sizes
- **Mechanistic insights**: Clear relationship between spectral radius and performance
- **Reproducible results**: Consistent benefits across multiple seeds

### **Stretch Success**
- **Strong statistical significance** (p < 0.01) with large effect sizes
- **Boundary complexity correlation**: Fractal dimension improvements
- **Criticality optimization**: Evidence of edge-of-chaos benefits
- **Generalization hints**: Benefits suggest broader applicability

## Implementation Strategy

### **Step 1**: Experimental Design *(Priority 1)*
```python
# Controlled A/B testing framework
experiments = {
    'baseline': {
        'config': 'configs/phase1_baseline.yaml',
        'regularization': None
    },
    'spectral': {
        'config': 'configs/phase1_spectral.yaml', 
        'regularization': 'FixedSpectralRegularizer(target_sigma=1.0)'
    }
}

# Identical conditions except regularization:
# - Same network architectures
# - Same optimizers and learning rates  
# - Same data splits and seeds
# - Same evaluation metrics
```

### **Step 2**: Training Efficiency Analysis *(Priority 2)*
```python
def analyze_training_efficiency(baseline_results, spectral_results):
    """Compare convergence speed and final performance"""
    # Epochs to target accuracy
    # Training loss convergence rate
    # Final boundary classification accuracy
    # Statistical significance testing
```

### **Step 3**: Criticality Correlation Analysis *(Priority 3)*
```python
def analyze_criticality_dynamics(results):
    """Correlate spectral radius with performance"""
    # Spectral radius evolution during training
    # Dead neuron rate trajectories
    # Boundary fractal dimension progression
    # Criticality score correlation with accuracy
```

## Experimental Protocol

### **Training Efficiency Metrics**
- **Epochs to 75% accuracy**: Primary efficiency measure
- **Loss convergence rate**: Secondary efficiency measure
- **Final accuracy after fixed epochs**: Quality measure
- **Training stability**: Variance across seeds

### **Boundary Learning Quality**
- **Classification accuracy**: Standard performance metric
- **Decision boundary fractal dimension**: Complexity measure
- **Robustness to perturbations**: Stability measure
- **Visual boundary quality**: Qualitative assessment

### **Criticality Indicators**
- **Spectral radius evolution**: Edge-of-chaos targeting
- **Dead neuron rate**: Network utilization
- **Perturbation sensitivity**: Critical dynamics
- **Unified criticality score**: Combined indicator

## Statistical Analysis Framework

### **Hypothesis Testing**
- **Null hypothesis**: No difference in training efficiency
- **Alternative hypothesis**: Spectral regularization affects performance-variance trade-offs
- **Test statistic**: Difference in epochs to convergence
- **Significance threshold**: p < 0.05

### **Effect Size Reporting**
- **Cohen's d**: Standardized effect size for efficiency difference
- **Confidence intervals**: 95% CI for all effect estimates
- **Practical significance**: Meaningful improvement thresholds
- **Power analysis**: Statistical power validation

## Engineering Patterns Application

**Reference**: [docs/engineering-patterns.md](./docs/engineering-patterns.md)

- **Identical controls**: Same everything except regularization
- **5+ seeds minimum**: Statistical rigor for all comparisons
- **Effect size reporting**: Confidence intervals, not just p-values
- **Real data throughout**: Belgium-Netherlands boundary focus

## Branch Workflow

```bash
git checkout -b plan2c-comparison
# Implement A/B testing experiments
# Execute baseline vs spectral comparisons
# Analyze results statistically
git checkout main && git merge plan2c-comparison
git branch -d plan2c-comparison
```

## Handover Preparation

**Completion triggers**:
- A/B experiments completed successfully
- Statistical analysis demonstrates spectral benefits (or lack thereof)
- Criticality correlation analysis complete
- Results ready for publication visualization

**Next context setup**:
- Update this document with COMPLETED status and key findings
- Create `docs/phase1-plan2d-visualization.md` for publication-quality figures
- Commit with experimental results and statistical validation

## Quality Gates

- [ ] Controlled A/B experiments executed
- [ ] Statistical significance testing completed
- [ ] Effect sizes quantified with confidence intervals
- [ ] Training efficiency differences documented
- [ ] Criticality indicators analyzed
- [ ] Results support or refute core SPECTRA hypothesis
- [ ] No scope creep beyond comparative experiments

**Context Budget Target**: Complete within single focused session (~50KB)