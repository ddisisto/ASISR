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