# SPECTRA Phase 1 Implementation: Belgium-Netherlands Boundary Learning

You are working on the Spectral Performance Control Through Regularization Analysis (SPECTRA) project - a cutting-edge research initiative investigating whether neural networks learn complex decision boundaries more efficiently when operating at the "edge of chaos" (spectral radius σ ≈ 1.0).

**Your Mission**: Implement Phase 1 proof-of-concept demonstrating spectral regularization benefits on the Belgium-Netherlands border classification task.

**Project Status**:
- Repository organized with clean architecture
- Documentation complete (CLAUDE.md, ARCHITECTURE.md, PROJECT_PLAN.md)
- Core theory validated by external conceptual analysis
- Ready for immediate implementation

**Current Focus**: Phase 1 - Boundary Mapping Proof-of-Concept
**Central Question**: *"Can spectral regularization at σ ≈ 1.0 enable networks to learn complex decision boundaries more efficiently than conventional training?"*

**Immediate Tasks**:
1. Migrate `prototypes/map_loader.py` → `asisr/data/map_loader.py` following ARCHITECTURE.md interfaces
2. Extract baseline MLP from `prototypes/SAMPLE-CODE-v1.md` → `asisr/models/mlp.py` with spectral hooks
3. Implement basic spectral regularization → `asisr/regularization/fixed.py` 
4. Create experiment comparing baseline vs. spectral on Belgium-Netherlands border learning
5. Generate publication-quality visualizations showing training dynamics and decision boundaries

**Critical Requirements**:
- Follow ARCHITECTURE.md plugin interfaces religiously - all components must implement abstract base classes
- Maintain scientific rigor: 5-seed minimum, statistical significance testing, proper error bars
- Implement incremental testing alongside each component - no untested code
- Focus on Phase 1 ONLY - do not implement adaptive features yet

**Success Criteria**:
- Networks successfully learn Belgium-Netherlands border classification  
- Clear visual demonstration that spectral regularization changes boundary learning
- Quantitative improvement in training efficiency (epochs to convergence)
- Statistical validation across multiple seeds

**Key Resources**:
- Technical authority: `ARCHITECTURE.md` - READ THIS for all structural decisions
- Research context: `PROJECT_PLAN.md` - READ THIS for scientific goals  
- Development workflow: `CLAUDE.md` - READ THIS for coding standards
- Legacy code: `prototypes/` - migrate from here following new architecture
- Dataset: `asisr/data/Baarle-Nassau_-_Baarle-Hertog-en.svg` - complex boundary map
- Theoretical validation: `docs/conceptual-analysis.md` - external expert review

**Development Principles**:
- Plugin architecture with abstract interfaces (SpectralRegularizedModel, SpectralRegularizer)
- Configuration-driven experiments with YAML files
- Comprehensive testing (unit + integration, >90% coverage target)
- Publication-quality visualization standards
- Complete reproducibility with seed management

**Warning**: Previous sub-agent created 60KB of unvalidated code when asked for "basic setup." Maintain scope discipline - implement incrementally with full oversight and testing.

**Repository Structure**:
```
SPECTRA/
├── CLAUDE.md, ARCHITECTURE.md, PROJECT_PLAN.md    # Authority documents
├── asisr/                                          # Main package (empty, ready for implementation)
├── prototypes/                                     # Legacy code to migrate  
├── docs/                                           # Research references
└── requirements.txt                                # Dependencies
```

Begin by reading ARCHITECTURE.md and PROJECT_PLAN.md, then start with the map loader migration. Each component must follow the defined interfaces and include comprehensive tests. The scientific integrity of this research depends on rigorous implementation standards.