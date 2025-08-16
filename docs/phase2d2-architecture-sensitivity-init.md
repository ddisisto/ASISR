# SPECTRA Phase 2D-2: Architecture Sensitivity Analysis

**Context**: Phase 2D-1 confirmed breakthrough (+1.04%, p=0.032) - now test generalizability  
**Mission**: Validate linear scheduling effect across micro to macro architectures  
**Timeline**: 2-3 days for rapid iteration  
**Branch**: `phase2d-breakthrough-validation`

## **Critical Research Question**

**Validated**: Linear σ scheduling shows +1.0% improvement on TwoMoons with [64,64] architecture  
**New Question**: Does this effect scale across architecture sizes, or is it capacity-dependent?

## **Architecture Sensitivity Hypothesis**

### **Potential Scaling Patterns**:
1. **Universal Effect**: Linear scheduling helps regardless of model size
2. **Capacity Threshold**: Effect emerges only above minimum complexity  
3. **Optimal Zone**: Effect peaks at specific capacity/regularization balance
4. **Overparameterization Penalty**: Large models hurt by constraining spectral radius

### **Speed Optimization Strategy**
**Key Insight**: Test micro-architectures first for rapid iteration
- **8x8, 16x16**: ~2-5 seconds per seed - can test extensively
- **32x32, 64x64**: ~10-20 seconds per seed - validate scaling  
- **128x128, 256x256**: ~30-60 seconds per seed - confirm at scale

## **Experimental Design**

### **Architecture Sweep** 
```yaml
# Complete architecture range for scaling analysis
architectures:
  micro:
    - [8, 8]         # Ultra-micro: 120 parameters
    - [16, 16]       # Micro: 464 parameters  
    - [32, 32]       # Small: 1,664 parameters
  
  standard:  
    - [64, 64]       # Phase 2B original: 5,888 parameters
    - [128, 128]     # Large: 21,760 parameters
    - [256, 256]     # Macro: 83,200 parameters
    
  alternative_shapes:
    - [16]           # Single layer: 48 parameters
    - [8, 16, 8]     # Deep narrow: 320 parameters
    - [32, 64, 32]   # Deep medium: 3,168 parameters
```

### **Reduced Seed Strategy for Speed**
**Micro-architectures**: 10 seeds each (sufficient for trend detection)  
**Standard architectures**: 5 seeds each (validation with known baseline)  
**Alternative shapes**: 5 seeds each (exploratory)

## **Execution Plan**

### **Phase 2D-2A: Micro-Architecture Rapid Testing** (Day 1)
**Target**: Complete [8,8], [16,16], [32,32] in <2 hours
```bash
# Ultra-fast micro tests - 10 seeds each
python run_experiment.py phase2b \
  --static configs/phase2d2_8x8_static_10seeds.yaml \
  --dynamic configs/phase2d2_8x8_linear_10seeds.yaml \
  --names Linear-8x8 \
  --plots

python run_experiment.py phase2b \
  --static configs/phase2d2_16x16_static_10seeds.yaml \
  --dynamic configs/phase2d2_16x16_linear_10seeds.yaml \
  --names Linear-16x16 \
  --plots

python run_experiment.py phase2b \
  --static configs/phase2d2_32x32_static_10seeds.yaml \
  --dynamic configs/phase2d2_32x32_linear_10seeds.yaml \
  --names Linear-32x32 \
  --plots
```

### **Phase 2D-2B: Standard Architecture Validation** (Day 2)
**Target**: Confirm [64,64] and test [128,128], [256,256]
```bash
# Validate known baseline + scale up
python run_experiment.py phase2b \
  --static configs/phase2d2_128x128_static_5seeds.yaml \
  --dynamic configs/phase2d2_128x128_linear_5seeds.yaml \
  --names Linear-128x128 \
  --plots

python run_experiment.py phase2b \
  --static configs/phase2d2_256x256_static_5seeds.yaml \
  --dynamic configs/phase2d2_256x256_linear_5seeds.yaml \
  --names Linear-256x256 \
  --plots
```

### **Phase 2D-2C: Alternative Shapes** (Day 3)
**Target**: Test different architectural patterns
```bash
# Single layer and deep narrow architectures
python run_experiment.py phase2b \
  --static configs/phase2d2_single_static_5seeds.yaml \
  --dynamic configs/phase2d2_single_linear_5seeds.yaml \
  --names Linear-Single \
  --plots

python run_experiment.py phase2b \
  --static configs/phase2d2_deep_static_5seeds.yaml \
  --dynamic configs/phase2d2_deep_linear_5seeds.yaml \
  --names Linear-Deep \
  --plots
```

## **Expected Scaling Patterns**

### **🟢 Universal Effect Pattern**
```
Architecture    Effect Size    p-value
[8,8]          +0.8-1.2%      < 0.05
[16,16]        +0.9-1.1%      < 0.05  
[32,32]        +1.0-1.2%      < 0.05
[64,64]        +1.0%          < 0.05 (confirmed)
[128,128]      +0.9-1.1%      < 0.05
[256,256]      +0.8-1.0%      < 0.05
```
**Implication**: Linear scheduling is fundamental optimization principle

### **🟡 Capacity Threshold Pattern**
```
Architecture    Effect Size    p-value
[8,8]          +0.1%          > 0.1 (no effect)
[16,16]        +0.4%          > 0.05 (marginal)
[32,32]        +0.8%          < 0.05 (emerging)
[64,64]        +1.0%          < 0.05 (confirmed)
[128,128]      +1.1%          < 0.01 (strong)
[256,256]      +1.2%          < 0.01 (strongest)
```
**Implication**: Effect scales with model expressivity

### **🔴 Architecture-Specific Artifact**
```
Architecture    Effect Size    p-value
[8,8]          -0.2%          > 0.1 (opposite)
[16,16]        +0.1%          > 0.1 (noise)
[32,32]        +0.3%          > 0.1 (weak)
[64,64]        +1.0%          < 0.05 (original)
[128,128]      +0.2%          > 0.1 (disappears)
[256,256]      -0.1%          > 0.1 (hurt)
```
**Implication**: Phase 2B was architecture-specific fluke

## **Decision Criteria**

### **Proceed to Phase 3** ✅
**Requirements**:
- **≥4/6 architectures** show p < 0.05
- **Effect size ≥0.5%** consistently  
- **No systematic architecture dependence**
- **Clear scaling trend** (monotonic or plateau)

### **Narrow Phase 3 Scope** ⚠️
**Pattern**:
- **Capacity-dependent**: Only works above/below threshold
- **Shape-dependent**: Works for specific width/depth ratios
- **Limited range**: Effective in narrow architecture window

### **Major Research Pivot** ❌  
**Evidence**:
- **<3/6 architectures** show significance
- **Inconsistent effect directions**
- **No clear scaling pattern**

## **Resource Optimization**

### **Computational Efficiency**
**Micro-architectures** (~2-5 seconds/seed):
- 10 seeds × 3 architectures × 2 configs = 60 micro-experiments (~5 minutes total)
- **Rapid trend detection** before investing in larger models

**Standard architectures** (~20-60 seconds/seed):  
- 5 seeds × 3 architectures × 2 configs = 30 standard experiments (~20 minutes total)
- **Validation of scaling** with known performance baseline

### **Early Termination**
**If micro-architectures show no effect**: Skip large architectures, conclude capacity-dependent  
**If inconsistent micro results**: Focus on replication rather than scaling
**If clear universal pattern**: Confirm with 1-2 large architectures only

## **Documentation Strategy**

### **Rapid Results Logging**
```bash
# Log all results to single file for cross-architecture analysis
echo "Architecture: [8,8] - Effect: +X.X% (p=Y.YY)" >> architecture_scaling_results.log
echo "Architecture: [16,16] - Effect: +X.X% (p=Y.YY)" >> architecture_scaling_results.log
# ... continue for all architectures
```

### **Scaling Visualization**
- **Effect size vs parameter count** scatter plot
- **Statistical significance vs architecture** heatmap  
- **Training dynamics comparison** across sizes

## **Success Metrics**

**Primary**: Determine architecture sensitivity pattern within 3 days  
**Secondary**: Establish computational efficiency for future experiments  
**Tertiary**: Identify optimal architecture range for Phase 3 research

---

**Bottom Line**: If linear scheduling works for 8x8 networks, we can iterate experiments in seconds instead of minutes. If it only works for specific sizes, we need to understand the capacity dependence before Phase 3.