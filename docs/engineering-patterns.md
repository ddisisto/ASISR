# ASISR Engineering Patterns

**Authority**: This document captures proven engineering patterns and quality standards validated during ASISR development. Reference this during implementation uncertainty.

## Proven Success Patterns *(Validated in Plan 1)*

### **Interface-Driven Development**
- **Plugin architecture discipline**: All components implement abstract base classes
- **No architectural drift**: Never deviate from ARCHITECTURE.md without updating it first
- **Separation of concerns**: Models, regularizers, metrics are completely independent
- **Extension points**: New components plug in without modifying existing code

### **Real Data Throughout**  
- **No synthetic fallbacks on critical path**: Use actual Belgium-Netherlands SVG data
- **SSoT maintenance**: Coordinate bounds `(-1.5, 2.5, -1.0, 1.5)` consistent everywhere
- **Tensor flow validation**: Real shapes from SVG → raster → coordinates → model → loss
- **Graceful degradation**: Handle missing dependencies without masking core functionality

### **Incremental Validation**
- **Component isolation**: Test each module independently before integration
- **Integration gates**: End-to-end validation before claiming success
- **Shape validation**: Extensive logging of tensor dimensions and device placement
- **Scope discipline**: Use `NotImplementedError("Feature X planned for Phase Y")` religiously
- **Honest test reporting**: Clearly distinguish tested vs deferred functionality in validation output

## Context Management Patterns *(Emerging)*

### **Micro-Commit Workflow**
- **Branch discipline**: Every feature gets branch → implement → test → merge → delete
- **Working main always**: Every merge leaves main in functional state
- **Frequent commits**: Preserve work before context compression
- **Clear commit messages**: Reference handover documents and completion status

### **Context Budget Management**
- **Monitor compression warnings**: UI shows "Context left until auto-compact: X%"
- **Handover triggers**: 12-15% context remaining OR natural completion points
- **Planning document updates**: Proactively update handover docs before limits
- **Scope gates**: Never start large features near context limits

### **Handover Protocol**
- **Living planning documents**: Tactical focus with universal patterns extracted
- **Git-based succession**: `git log -1` + handover document discovery
- **Status preservation**: Clear current/completed/pending sections
- **Pattern reference**: Point to this document for engineering guidance

## Statistical Rigor Framework *(Universal)*

### **Experimental Design**
- **5+ seeds minimum** for all comparative claims
- **Configuration-driven**: YAML files for all experimental parameters  
- **Identical controls**: Same architectures, optimizers, data splits
- **Reproducibility**: Deterministic seed management across numpy/torch/random

### **Statistical Analysis**
- **Effect size reporting**: Confidence intervals, not just p-values
- **Significance testing**: Proper statistical validation of improvements
- **Error bars everywhere**: All plots include uncertainty quantification
- **Publication standards**: Professional figure quality from day one

### **Documentation Standards**
- **Version-controlled configs**: All experiment parameters in git
- **Comprehensive logging**: Metrics collection and storage automation
- **Automated pipelines**: Figure generation with consistent styling
- **Reproducibility validation**: Results verified across different environments

## Quality Gates *(Universal)*

### **Implementation Quality**
- **Type hints mandatory**: All functions fully annotated
- **Comprehensive docstrings**: All public interfaces documented
- **Error handling**: Graceful failures with informative messages
- **Performance awareness**: <10% overhead vs baseline training

### **Scientific Quality**
- **No claims without statistics**: Quantitative validation required
- **Boundary analysis validation**: Fractal dimensions match expected complexity
- **Training dynamics monitoring**: Real-time criticality assessment
- **Comparative rigor**: Head-to-head experiments with proper controls

### **Code Organization**
- **Package structure**: Follow ARCHITECTURE.md plugin patterns
- **Import cleanliness**: Proper `__all__` declarations and dependency management
- **Testing symmetry**: Test structure mirrors package structure  
- **Configuration management**: YAML-driven with schema validation

## Anti-Patterns *(Avoid These)*

### **Scope Creep**
- ❌ **Feature mixing**: Implementing across phases simultaneously
- ❌ **Architecture deviation**: Changing structure without documentation updates
- ❌ **Mock data**: Synthetic datasets masking real loading issues
- ❌ **Context overreach**: Starting large features near limits

### **Documentation Drift**
- ❌ **Outdated references**: File paths not matching reorganized structure
- ❌ **Authority confusion**: Multiple documents claiming same scope
- ❌ **Pattern scatter**: Engineering insights buried in tactical documents
- ❌ **Over-documentation**: Death by analysis paralysis

### **Statistical Shortcuts**
- ❌ **Single-seed claims**: Unreproducible experimental results
- ❌ **P-value hunting**: Statistical significance without effect sizes
- ❌ **Cherry-picking**: Selective result reporting
- ❌ **Baseline skipping**: Claims without proper control comparisons

### **Testing Anti-Patterns** *(Critical)*
- ❌ **Mock data validation**: Synthetic test data that masks real integration issues
- ❌ **False positive reporting**: Tests that appear to pass but don't validate actual functionality
- ❌ **Premature success claims**: Marking features as "working" before end-to-end validation
- ❌ **Interface-only testing**: Testing that functions can be imported without testing they work correctly

## Pattern Evolution

This document grows with project wisdom. Add new patterns as they emerge, but maintain focus on **universally applicable insights** rather than tactical implementation details.

**Update triggers**:
- Major technical breakthrough or failure
- Repeated debugging patterns
- Context management innovations  
- Statistical analysis improvements

**Quality standards**:
- Patterns must be validated through actual use
- Focus on actionable guidelines, not abstract principles
- Maintain conciseness - delete obsolete patterns
- Reference from CLAUDE.md as engineering authority