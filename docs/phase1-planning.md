# Phase 1 Implementation Planning

## Plan 1: Foundation to First Git Commit

### **Core Implementation Steps**
1. **Foundation**: Implement base classes (`SpectralRegularizedModel`, `SpectralRegularizer`, `CriticalityMonitor`) per ARCHITECTURE.md
2. **Data Migration**: Adapt `prototypes/map_loader.py` → `asisr/data/map_loader.py` with interface compliance
3. **Model Architecture**: Extract MLP from sample code → `asisr/models/mlp.py` implementing spectral interfaces
4. **Regularization**: Create `asisr/regularization/fixed.py` for σ=1.0 targeting
5. **Integration Test**: Basic Belgium-Netherlands boundary learning experiment

### **SSoT Validation Strategy**
- Coordinate bounds consistency: `(-1.5, 2.5, -1.0, 1.5)` across all components
- Interface compliance: All implementations satisfy abstract base classes
- Tensor shape validation: Data loader → model → regularizer flow works
- Unit tests for each component before integration

### **Risk Mitigation**
- Test components in isolation first
- Use minimal datasets initially (50x50 grids)
- Extensive logging for tensor shapes and device placement
- Graceful fallbacks for missing dependencies

### **Success Criteria**
- **Minimum**: All interfaces implemented, basic integration test runs
- **Target**: Belgium-Netherlands boundary learning with visual baseline/spectral differences
- **Stretch**: Multi-seed statistical validation
- **Scope Maintained**: Nothing added beyond plan. Judicious use of NotImplementedError. No mock/placeholder data on critical path.

### **Implementation Discipline**
- Use `raise NotImplementedError("Feature X planned for Phase Y")` for unimplemented features
- Never create synthetic/mock data that could mask real data loading issues
- Maintain strict scope boundaries - implement only what's needed for this commit
- Save advanced features (adaptive regularization, multi-scale) for future phases

This establishes the plugin architecture foundation with validated components ready for Phase 1 experimentation.

---

## Plan 2: Scientific Validation to Phase 1 Completion

### **Strategic Context** *(Building on Plan 1 Success)*
Plan 1 delivered a rock-solid foundation with working plugin architecture. The transition from design to working system is complete. Plan 2 focuses on scientific rigor and experimental validation to complete Phase 1 objectives.

**Core Research Question**: *"Can spectral regularization at σ ≈ 1.0 enable networks to learn complex decision boundaries more efficiently than conventional training?"*

### **Implementation Priorities**
1. **Complete Criticality Implementation**: Full boundary fractal dimension analysis from prototypes/SAMPLE-CODE-v1.md
2. **Multi-Seed Statistical Framework**: 5+ seed experimental validation with proper significance testing
3. **Baseline vs Spectral Comparison**: Head-to-head experiments demonstrating spectral regularization benefits
4. **Publication-Quality Visualization**: Decision boundary plots, training dynamics, statistical analysis

### **Technical Implementation Steps**

#### **Step 1: Complete Criticality Monitoring**
**Goal**: Implement full boundary fractal dimension analysis
**Files**: 
- `asisr/metrics/criticality.py` - Complete `_compute_boundary_fractal_dim()` 
- `asisr/visualization/boundaries.py` - Boundary extraction and visualization
**Implementation**: Extract box-counting algorithm from prototypes/SAMPLE-CODE-v1.md
**Validation**: Fractal dimension computation matches expected complexity patterns

#### **Step 2: Multi-Seed Experimental Framework**
**Goal**: Scientific rigor with statistical validation
**Files**:
- `experiments/phase1_boundary_mapping/multi_seed_experiment.py` - Statistical experiment runner
- `asisr/training/experiment.py` - Formal experiment orchestration
- `configs/phase1_baseline.yaml`, `configs/phase1_spectral.yaml` - Configuration management
**Implementation**: YAML-driven experiments with seed management and result aggregation
**Validation**: Reproducible results across multiple runs with proper error bars

#### **Step 3: Baseline vs Spectral Comparison**
**Goal**: Demonstrate spectral regularization benefits
**Implementation**:
- Controlled A/B testing framework
- Identical architectures with/without spectral regularization
- Training efficiency metrics (epochs to convergence)
- Decision boundary quality assessment
**Validation**: Statistically significant improvement in training dynamics

#### **Step 4: Publication-Quality Visualization**
**Goal**: Professional presentation of results
**Files**:
- `asisr/visualization/dynamics.py` - Training trajectory plots
- `asisr/visualization/boundaries.py` - Decision boundary comparison
- `experiments/phase1_boundary_mapping/generate_figures.py` - Figure generation pipeline
**Implementation**: matplotlib/seaborn publication standards with error bars
**Validation**: Publication-ready figures demonstrating spectral regularization benefits

### **Scientific Rigor Framework**

#### **Statistical Validation Requirements**
- **Minimum 5 seeds** for all comparative claims
- **Effect size reporting** with confidence intervals  
- **Statistical significance testing** (not just p-values)
- **Reproducibility validation** with deterministic seed management

#### **Experimental Controls**
- **Architecture consistency**: Identical networks except for regularization
- **Training consistency**: Same optimizers, learning rates, batch sizes
- **Data consistency**: Same train/test splits across all experiments
- **Evaluation consistency**: Standardized metrics across all runs

#### **Documentation Standards**
- **Configuration files**: All experiment parameters in version control
- **Result logging**: Comprehensive metrics collection and storage
- **Figure generation**: Automated pipeline with consistent styling
- **Statistical reporting**: Proper error bars and significance testing

### **Success Criteria Evolution**

#### **Minimum Success (Phase 1 Complete)**
- Full criticality monitoring implementation working
- Multi-seed experiments demonstrate reproducible results
- Clear visual evidence of spectral regularization effects
- Statistical validation framework operational

#### **Target Success (Strong Phase 1)**
- **Quantitative improvement**: 10-20% training efficiency gain with spectral regularization
- **Boundary quality**: Measurable improvement in decision boundary learning
- **Criticality correlation**: Evidence linking spectral radius to boundary complexity
- **Publication-ready results**: Professional figures and statistical analysis

#### **Stretch Success (Exceptional Phase 1)**
- **Strong statistical significance**: p < 0.01 with substantial effect sizes
- **Mechanistic insights**: Clear understanding of why spectral regularization helps
- **Generalization evidence**: Benefits extend beyond single boundary map
- **Theoretical validation**: Results support SPECTRA hypothesis predictions

### **Risk Management & Scope Control**

#### **Technical Risks**
- **Boundary analysis complexity**: Box-counting implementation challenges
- **Statistical power**: Insufficient effect size for significance
- **Visualization quality**: Professional figure generation difficulties
- **Performance overhead**: Spectral regularization computational cost

#### **Scope Discipline**
- **Phase 1 only**: No adaptive regularization (Phase 2)
- **Single dataset**: Belgium-Netherlands boundary focus
- **Fixed architectures**: MLP only, no CNNs/Transformers
- **Core metrics**: Focus on training efficiency and boundary quality

#### **Quality Gates**
- Each component tested independently before integration
- Statistical significance required before claiming improvements
- Publication standards enforced for all figures
- Reproducibility validated across different environments

### **Strategic Patterns from Plan 1 to Maintain**

#### **Successful Development Patterns**
- **Interface-driven development**: Maintain plugin architecture discipline
- **Real data throughout**: No synthetic fallbacks on critical path
- **Incremental validation**: Test each component before integration
- **SSoT maintenance**: Consistent coordinate systems and data formats

#### **Project Management Patterns**
- **Clear scope boundaries**: Use NotImplementedError for future phases
- **Commit discipline**: Frequent commits with clear success criteria
- **Documentation first**: Update planning docs before implementation
- **User feedback loops**: Regular check-ins on direction and priorities

This plan builds systematically on Plan 1's foundation to deliver scientifically rigorous Phase 1 completion with publication-quality results.

---

## Plan 2 Implementation: Micro-Phases for Context Management

**Context Budget Crisis Resolution**: Plan 2 as originally written (~130 lines) exceeds single context capacity. Split into focused sub-phases with clear handover points.

### **Phase Execution Order**
1. **Plan 2-A**: [Criticality Implementation](./phase1-plan2a-criticality.md) (~30KB)
2. **Plan 2-B**: [Multi-Seed Framework](./phase1-plan2b-multiseed.md) (~40KB)  
3. **Plan 2-C**: [Baseline vs Spectral Comparison](./phase1-plan2c-comparison.md) (~50KB)
4. **Plan 2-D**: [Publication Visualization](./phase1-plan2d-visualization.md) (~40KB)

### **Engineering Patterns Authority**
Universal patterns extracted to: [docs/engineering-patterns.md](./engineering-patterns.md)

### **Handover Protocol**
- Each sub-phase gets dedicated context
- Branch workflow: create → implement → test → merge → delete
- Update planning docs before context handover
- Reference patterns document for engineering guidance

**Phase 1 completion**: All sub-phases complete with publication-ready results.