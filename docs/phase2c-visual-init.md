# SPECTRA Phase 2C: Visual Exploration Framework

**Context**: Visual-Dynamics-Explorer phase following Phase 2B scientific breakthrough  
**Mission**: Transform Phase 2B statistical validation into visual understanding and interactive exploration

## **Phase 2B Foundation** ✅ **VALIDATED**

**Scientific Breakthrough Achieved**:
- **Linear σ Schedule**: +1.1% accuracy improvement (p=0.0344*, d=1.610 large effect)
- **Exponential Schedule**: +0.7% improvement (p=0.1447, d=1.022 large effect)  
- **Training-phase σ control**: Hypothesis confirmed with statistical rigor

**Visual Assets Created**:
- `phase2b_comparison.png`: Publication-quality performance comparison ✅
- `phase2b_dynamics.png`: Framework exists but **trajectory data missing** ⚠️

## **Phase 2C Objectives**

### **Primary Goal**: Bridge from statistical validation to visual understanding

**Core Research Questions**:
1. **What do σ schedules actually look like?** (σ(t) evolution curves)
2. **How do different strategies affect training dynamics?** (real-time trajectories)  
3. **What's the optimal parameter space?** (interactive exploration)
4. **How can we make this explorable?** (parameter sensitivity surfaces)

### **Technical Milestones**:

**Milestone 1: Fix Trajectory Data Gap** 🔧
- Modify experiment framework to collect epoch-by-epoch metrics
- Generate actual training dynamics curves (the missing plots)
- Show σ evolution overlaid on performance trajectories

**Milestone 2: σ Schedule Visualization Suite** 📊  
- Interactive σ(t) curve generation for all strategies
- Parameter sliders: initial_σ, final_σ, decay_rate, schedule_type
- Real-time schedule preview and comparison

**Milestone 3: Training Dynamics Dashboard** 🎯
- Live training visualization with σ evolution overlay
- Performance-variance surface mapping
- Multi-strategy comparison interface

**Milestone 4: Publication-Quality Interactive System** ✨
- Export-ready figure generation
- Parameter sensitivity exploration
- Interactive scientific storytelling interface

## **Immediate Development Tasks**

### **Task 1: Trajectory Data Collection** (Critical)
**Problem**: Current experiments only save final metrics, dynamics plots show "data not available"
**Solution**: Extend `SPECTRAExperiment` class to log epoch trajectories
**Files**: `spectra/training/experiment.py`
**Outcome**: Enable actual training dynamics visualization

### **Task 2: σ Schedule Plotting Utilities** 
**Need**: Visualize what different scheduling algorithms actually do
**Implementation**: 
```python
# spectra/visualization/schedules.py
def plot_sigma_schedules(strategies, epochs=100):
    # Show σ(t) curves for Linear, Exponential, Step
    # Interactive parameter controls
    # Side-by-side comparison
```

### **Task 3: Enhanced Dynamics Plots**
**Goal**: Replace "trajectory data not available" with actual curves
**Show**: 
- Accuracy evolution with σ(t) overlay
- Spectral radius tracking (actual vs target)
- Criticality score progression
- Boundary complexity changes

## **Visual Appeal Strategy**

### **Phase 2C-A: Foundation** (publication ready)
- **Clean trajectory plots** showing mechanism of dynamic control
- **σ schedule gallery** comparing all strategies visually
- **Performance evolution** with statistical confidence bounds

### **Phase 2C-B: Interactivity** (exploration ready)  
- **Parameter sliders** for real-time schedule adjustment
- **Click-to-explore** parameter sensitivity
- **Animated training progression** showing σ evolution

### **Phase 2C-C: Dashboard** (presentation ready)
- **Integrated comparison suite** for static vs dynamic
- **Export functionality** for publication figures  
- **Scientific storytelling** interface

## **Technical Architecture**

**Build on existing**:
- ✅ `spectra/regularization/dynamic.py` - Scheduling algorithms validated
- ✅ `experiments/phase2b_dynamic_spectral/` - Comparison framework proven
- 🔄 `spectra/visualization/` - Minimal, needs major development

**New components needed**:
- `spectra/visualization/schedules.py` - σ(t) curve generation and plotting
- `spectra/visualization/dynamics.py` - Training trajectory visualization  
- `spectra/visualization/interactive.py` - Parameter exploration interface
- `spectra/visualization/dashboard.py` - Integrated comparison system

## **Success Criteria**

**Minimum Viable Visual Layer**:
- ✅ Trajectory data collection working
- ✅ σ schedule visualization implemented
- ✅ Training dynamics plots functional (replace current empty plots)

**Target Visual System**:
- ✅ Interactive parameter exploration
- ✅ Real-time schedule preview
- ✅ Publication-quality figure export

**Stretch Visual Dashboard**:
- ✅ Web-based interactive interface
- ✅ Parameter sensitivity surfaces
- ✅ Animated training progression

## **Development Approach**

**Branch Strategy**: Create `phase2c-visual-framework` for development
**Integration**: Build incrementally, test with existing Phase 2B data
**Validation**: Generate enhanced versions of current plots first
**Extension**: Add interactivity once foundation solid

**Git Discipline**: Sacred - every visual component properly tested and documented

---

**Ready for Visual Exploration**: Phase 2B scientific foundation is rock-solid. Time to make the mechanism visible and explorable! 🎨