# SPECTRA Phase 3: Adaptive Spectral Optimization

**Context**: Phase 2D capacity threshold discovery revolutionizes spectral scheduling approach  
**Mission**: Develop universal adaptive optimization framework based on capacity-complexity matching  
**Timeline**: 4-6 weeks  
**Branch**: `phase3-adaptive-optimization`

## **Scientific Foundation**

### **Phase 2D Breakthrough: Capacity Threshold Effect**
**Core Discovery**: Linear spectral scheduling effectiveness depends on **network capacity relative to problem complexity**

| Architecture | Parameters | Linear Effect | Interpretation |
|-------------|------------|---------------|----------------|
| 8x8         | ~120       | **-1.1%**     | Under-parameterized: hurt by exploration phase |
| 16x16       | ~464       | **+2.0%**     | Optimal capacity: perfect exploration-exploitation |
| 32x32       | ~1,664     | **+2.2%**     | Peak effectiveness: ideal regularization balance |
| 64x64       | ~5,888     | **+1.0%**     | Over-parameterized: regularization less critical |

### **Unified Theory Hypothesis**
**Capacity-Complexity Matching**: Optimal spectral scheduling must adapt σ trajectories based on:
1. **Network capacity** (parameter count, architecture)
2. **Problem complexity** (boundary complexity, dataset difficulty)  
3. **Training dynamics** (gradient flows, spectral radius evolution)

## **Phase 3 Research Questions**

### **Primary Questions**
1. **Adaptive Scheduling**: Can we develop σ trajectories that eliminate negative effects for small networks while preserving benefits for larger ones?
2. **Universal Framework**: What mathematical relationship governs optimal σ scheduling across all capacity-complexity combinations?
3. **Predictive Theory**: Can we predict the optimal architecture size for a given problem complexity?

### **Deep Scientific Questions**
1. **Capacity Threshold Law**: What is the mathematical relationship between parameter count and optimal σ trajectories?
2. **Cross-Dataset Validation**: Does capacity-adaptive scheduling explain and resolve the Phase 2C boundary complexity correlation?
3. **Architecture Design**: Can we derive principled guidelines for selecting network sizes based on problem characteristics?

## **Experimental Design**

### **Phase 3A: Capacity-Adaptive Scheduling (Weeks 1-2)**

#### **Hypothesis**: Capacity-dependent initial σ values enable universal improvements

**Mathematical Framework**:
```python
σ_initial(capacity_ratio) = σ_base * (capacity_ratio)^β
σ_schedule(t, capacity) = σ_initial(capacity) * linear_decay(t)

where:
- capacity_ratio = actual_params / optimal_params
- β ∈ [-0.5, -0.2, 0.0, 0.2, 0.5] (parameter to optimize)
- σ_base = 2.5 (baseline from Phase 2B)
```

#### **Experimental Protocol**:
1. **Architecture Sweep**: Test 8x8, 16x16, 32x32, 64x64, 128x128 on TwoMoons
2. **Capacity Scaling**: Vary β parameter to find optimal capacity dependence
3. **Statistical Validation**: 10 seeds per configuration for robust results
4. **Success Metric**: All architectures show positive improvements

#### **Expected Outcomes**:
- **β < 0**: Smaller networks get higher initial σ (more exploration time)
- **β = 0**: Capacity-independent (current linear schedule)
- **β > 0**: Larger networks get higher initial σ (surprising if true)

### **Phase 3B: Cross-Dataset Generalization (Weeks 3-4)**

#### **Hypothesis**: Capacity-adaptive scheduling resolves Phase 2C boundary complexity issues

**Test Protocol**:
1. **Apply optimal capacity scheduling** from Phase 3A to Belgium-Netherlands and Circles datasets
2. **Architecture sensitivity** across all three datasets with adaptive scheduling
3. **Compare against Phase 2C results** to validate universal improvement

**Target Results**:
- **Belgium-Netherlands**: Eliminate -2.5% decrease, achieve positive improvement
- **Circles**: Eliminate -0.2% decrease, achieve positive improvement  
- **TwoMoons**: Maintain or improve +1.0% effect across all architectures

### **Phase 3C: Theoretical Framework (Weeks 5-6)**

#### **Mathematical Modeling**
**Derive capacity-complexity matching theory**:

```python
optimal_architecture_size = f(boundary_complexity, dataset_size, target_performance)
optimal_σ_schedule = g(architecture_size, problem_complexity, training_phase)
```

#### **Validation Experiments**:
1. **Predict optimal architectures** for each dataset using derived theory
2. **Test predictions** experimentally with capacity-adaptive scheduling
3. **Develop design guidelines** for architecture selection

## **Technical Implementation**

### **Adaptive Regularizer Implementation**
```python
class CapacityAdaptiveSpectralRegularizer(SpectralRegularizer):
    def __init__(self, model, capacity_ratio, beta=-0.2):
        self.model = model
        self.capacity_ratio = capacity_ratio
        self.beta = beta
        self.sigma_base = 2.5
        
    def get_sigma_schedule(self, epoch, total_epochs):
        sigma_initial = self.sigma_base * (self.capacity_ratio ** self.beta)
        decay_factor = 1.0 - (epoch / total_epochs)
        return sigma_initial * decay_factor + 1.0  # → 1.0 by end
```

### **Capacity Calculation**
```python
def calculate_capacity_ratio(model, dataset_complexity):
    actual_params = sum(p.numel() for p in model.parameters())
    
    # Empirically derived from Phase 2D results
    optimal_params_map = {
        'TwoMoons': 464,      # 16x16 architecture optimal
        'Circles': 1664,      # 32x32 architecture (predicted)
        'Belgium': 5888,      # 64x64 architecture (predicted)
    }
    
    optimal_params = optimal_params_map.get(dataset_complexity, 1664)
    return actual_params / optimal_params
```

## **Success Criteria**

### **Minimum Success** ✅
- **8x8 networks**: Eliminate negative effects, achieve ≥0% improvement
- **All architectures**: Consistent positive improvements on TwoMoons
- **Theoretical understanding**: Clear capacity-complexity relationship identified

### **Target Success** 🎯
- **Universal improvements**: +1-2% across all architectures and datasets
- **Predictive power**: Theory correctly predicts optimal architectures
- **Cross-dataset validation**: Belgium and Circles show positive improvements

### **Stretch Success** 🚀
- **Unified theory**: Mathematical framework for capacity-complexity optimization
- **Design principles**: Actionable guidelines for architecture selection
- **Publication-ready**: Results suitable for top-tier machine learning conference

## **Risk Mitigation**

### **Technical Risks**
- **Implementation complexity**: Start with simple capacity scaling, iterate
- **Hyperparameter sensitivity**: Systematic grid search for β parameter
- **Computational cost**: Use 50-epoch experiments for rapid iteration

### **Scientific Risks**
- **Theory doesn't generalize**: Fall back to empirical capacity-specific scheduling
- **Cross-dataset failure**: Focus on architecture design principles instead
- **No universal improvement**: Document capacity-dependent optimization as contribution

## **Experimental Timeline**

### **Week 1: Foundation**
- Implement CapacityAdaptiveSpectralRegularizer
- Test β parameter sweep on TwoMoons across all architectures
- Identify optimal capacity dependence

### **Week 2: Validation**
- Validate optimal β across multiple seeds and architectures
- Test edge cases (very small/large networks)
- Document capacity-adaptive scheduling framework

### **Week 3: Cross-Dataset**
- Apply optimal scheduling to Belgium-Netherlands dataset
- Test Circles dataset with capacity-adaptive approach
- Compare against Phase 2C baseline results

### **Week 4: Generalization**
- Architecture sensitivity across all three datasets
- Statistical validation of universal improvements
- Document cross-dataset effectiveness

### **Week 5: Theory**
- Develop mathematical framework for capacity-complexity matching
- Derive optimal architecture prediction equations
- Test theoretical predictions experimentally

### **Week 6: Publication**
- Comprehensive results analysis and visualization
- Draft publication-quality experimental section
- Document design principles and guidelines

## **Deliverables**

### **Code Deliverables**
- `CapacityAdaptiveSpectralRegularizer` implementation
- Configuration files for all Phase 3 experiments
- Analysis scripts for capacity-complexity relationships

### **Scientific Deliverables**
- Capacity threshold theory and mathematical framework
- Cross-dataset validation of universal optimization
- Architecture design principles based on problem complexity

### **Documentation Deliverables**
- Complete experimental protocol and results
- Theoretical derivations and proofs
- Practical guidelines for applying adaptive spectral optimization

---

**Bottom Line**: Phase 2D's capacity threshold discovery transforms spectral scheduling from a fixed methodology into an adaptive optimization framework. Phase 3 will develop this into a universal principle for neural network training, potentially revolutionizing how we think about architecture selection and regularization.