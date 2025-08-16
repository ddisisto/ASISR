# SPECTRA Architecture Debt Analysis: Phase Naming Decoupling

**Issue**: Experiment naming and output paths hardcoded to specific phases, causing organizational debt as project evolves.

## **Current Problems**

### **1. Hardcoded Phase Dependencies**
```python
# run_experiment.py - PROBLEMATIC
phase2b_dir = self.output_base / "phase2b"  # Hardcoded!
comparison_plot = phase2b_dir / f"phase2b_comparison_{strategy_name}.png"
```

### **2. Mixed Phase Outputs**
- Phase 2D experiments save to `plots/phase2b/` directories
- Configuration names vs output paths inconsistent
- Legacy experiment classes tied to specific phase naming

### **3. Tight Coupling Issues**
- Experiment runners know specific phase details
- Output path generation not configurable
- Phase transitions require manual code updates

## **Proposed Solutions**

### **Solution 1: Configuration-Driven Output Paths** ⭐
**Best for: Immediate improvement with minimal disruption**

```python
# In experiment config files
experiment:
  name: "phase3a_beta_sweep" 
  phase: "phase3a"              # NEW: Explicit phase designation
  output_base: "phase3a"        # NEW: Configurable output directory
  
# Experiment framework automatically uses:
# plots/{output_base}/{experiment_name}/
```

### **Solution 2: Phase-Aware Experiment Framework** ⭐⭐
**Best for: Long-term maintainability**

```python
class PhaseManager:
    """Manages phase-specific naming and output organization."""
    
    def __init__(self, phase_name: str):
        self.phase_name = phase_name
        self.output_base = Path("plots") / phase_name
    
    def get_experiment_dir(self, experiment_name: str) -> Path:
        return self.output_base / experiment_name
    
    def get_plot_path(self, experiment_name: str, plot_type: str) -> Path:
        return self.get_experiment_dir(experiment_name) / f"{plot_type}_{experiment_name}.png"

# Usage in experiments
phase_manager = PhaseManager.from_config(config)
output_dir = phase_manager.get_experiment_dir(config.experiment['name'])
```

### **Solution 3: Abstract Experiment Interface** ⭐⭐⭐
**Best for: Complete decoupling and future-proofing**

```python
class ExperimentExecutor(ABC):
    """Phase-agnostic experiment execution interface."""
    
    @abstractmethod
    def get_output_strategy(self) -> OutputStrategy:
        """Return strategy for managing experiment outputs."""
        pass
    
    @abstractmethod  
    def get_naming_strategy(self) -> NamingStrategy:
        """Return strategy for experiment naming."""
        pass

class Phase3Executor(ExperimentExecutor):
    """Phase 3 specific experiment execution."""
    
    def get_output_strategy(self) -> OutputStrategy:
        return CapacityAdaptiveOutputStrategy()
        
    def get_naming_strategy(self) -> NamingStrategy:
        return SemanticNamingStrategy()  # beta_sweep_8x8, not phase3a_something
```

## **Immediate Recommendations**

### **Priority 1: Configuration-Driven Paths** (1-2 hours)
1. Add `output_base` field to experiment configs
2. Update experiment framework to use configurable paths
3. Ensure Phase 3 experiments use proper phase3 directories

### **Priority 2: Standardize Existing Outputs** (30 minutes)
1. Add output_base to all Phase 3 configs: `output_base: "phase3a"`, etc.
2. Validate that new experiments create phase-appropriate directories
3. Document the transition in ARCHITECTURE.md

### **Priority 3: Legacy Cleanup** (Future)
1. Migrate Phase 2 experiments to use new system
2. Implement abstract experiment interfaces
3. Create phase-specific output strategies

## **Implementation Plan**

### **Step 1: Enhance Configuration Schema**
```yaml
# Enhanced experiment config schema
experiment:
  name: "beta_sweep_8x8"          # Semantic name
  phase: "phase3a"                # Phase identifier  
  output_base: "phase3a"          # Output directory
  description: "..."
```

### **Step 2: Update Experiment Framework**
```python
def get_output_directory(self) -> Path:
    """Get experiment output directory from configuration."""
    phase = self.config.experiment.get('phase', 'unknown')
    output_base = self.config.experiment.get('output_base', phase)
    experiment_name = self.config.experiment['name']
    
    return Path("plots") / output_base / experiment_name
```

### **Step 3: Validation & Testing**
- Ensure all Phase 3 configs specify correct output paths
- Validate no conflicts with existing Phase 2 outputs
- Test that framework creates proper directory structures

## **Benefits of Decoupling**

1. **Phase Independence**: Experiments not tied to specific phase naming
2. **Flexible Organization**: Output paths configurable per experiment
3. **Clean Transitions**: New phases don't require code changes
4. **Semantic Naming**: Experiment names describe functionality, not phase
5. **Future-Proof**: Architecture scales to Phase 4, 5, and beyond

## **Risk Mitigation**

- **Backward Compatibility**: Existing Phase 2 experiments continue working
- **Gradual Migration**: Can implement incrementally without breaking changes  
- **Default Fallbacks**: Framework provides sensible defaults if config incomplete
- **Clear Documentation**: Update patterns documented in ARCHITECTURE.md

---

**Recommendation**: Implement **Solution 1** immediately for Phase 3, then evolve toward **Solution 2** for long-term maintainability.