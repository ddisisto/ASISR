# Phase 1 Plan 2-A: Complete Criticality Implementation

**Sub-phase Focus**: Implement full boundary fractal dimension analysis to complete criticality monitoring system.

**Context Budget**: ~30KB (single focused implementation)

**Dependencies**: Plan 1 foundation complete ✅

## Objective

Complete the criticality monitoring system by implementing boundary fractal dimension analysis from `prototypes/SAMPLE-CODE-v1.md`. This will enable full `assess_criticality()` functionality instead of the current private method fallbacks.

## Implementation Scope

### **Primary Target**
- `spectra/metrics/criticality.py` - Complete `_compute_boundary_fractal_dim()` method
- Extract box-counting algorithm from prototype code
- Integrate with existing criticality framework

### **Secondary Targets** 
- `spectra/visualization/boundaries.py` - Basic boundary extraction for fractal analysis
- Test validation of fractal dimension computation
- Integration test updates to use full criticality assessment

### **Strict Scope Boundaries**
- ❌ **No multi-seed experiments** (Plan 2-B)
- ❌ **No A/B testing framework** (Plan 2-C)  
- ❌ **No publication visualization** (Plan 2-D)
- ❌ **No configuration management** (Plan 2-B)

## Success Criteria

### **Minimum Success**
- `CriticalityMonitor.assess_criticality()` works without private method fallbacks
- Boundary fractal dimension computation produces reasonable values
- Integration test runs without NotImplementedError exceptions

### **Target Success**  
- Fractal dimension correlates with expected boundary complexity
- Full criticality score combines all metrics appropriately
- Basic boundary visualization supports analysis

## Implementation Strategy

### **Step 1**: Extract Fractal Analysis *(Priority 1)*
```python
# From prototypes/SAMPLE-CODE-v1.md
def _compute_boundary_fractal_dim(self, model, data, resolution=100):
    """Extract and adapt box-counting algorithm"""
    # Decision boundary extraction
    # Box-counting implementation  
    # Fractal dimension calculation
```

### **Step 2**: Complete Integration *(Priority 2)*
```python
def assess_criticality(self, model, data):
    """Full implementation without private method fallbacks"""
    # Dead neuron rate ✅ (already implemented)
    # Perturbation sensitivity ✅ (already implemented) 
    # Boundary fractal dimension ← implement this
    # Spectral radius ✅ (already implemented)
```

### **Step 3**: Validation Testing *(Priority 3)*
- Unit tests for box-counting algorithm
- Integration test updates using full criticality
- Fractal dimension sanity checks

## Engineering Patterns Application

**Reference**: [docs/engineering-patterns.md](./engineering-patterns.md)

- **Interface-driven**: Maintain `CriticalityMonitor` interface compatibility
- **Real data throughout**: Use actual Belgium-Netherlands boundary for fractal analysis
- **Incremental validation**: Test fractal computation independently before integration
- **Scope discipline**: Resist temptation to add visualization features beyond basic needs

## Branch Workflow

```bash
git checkout -b plan2a-criticality
# Implement criticality completion
# Test thoroughly
# Commit frequently
git checkout main && git merge plan2a-criticality
git branch -d plan2a-criticality
```

## Handover Preparation

**Completion triggers**:
- Full criticality assessment working
- Integration tests pass without NotImplementedError
- Basic fractal dimension validation complete

**Next context setup**:
- Update this document with COMPLETED status
- Create `docs/phase1-plan2b-multiseed.md` for next sub-phase
- Commit with clear handover reference

## Quality Gates

- [ ] Box-counting algorithm extracted and working
- [ ] Fractal dimension produces reasonable values (0.5-2.0 range expected)
- [ ] Integration test uses full criticality without private method calls
- [ ] No scope creep beyond criticality implementation
- [ ] All changes committed to working main branch

**Context Budget Target**: Complete within single focused session (~30KB)