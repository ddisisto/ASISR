# CUDA Fix Handover: Enable Larger Scale Experiments

**Date**: August 16, 2025  
**Context**: SPECTRA Phase 3 complete, need CUDA compatibility for meaningful effect detection  
**Branch**: `phase3-adaptive-optimization`  
**Handover Scope**: CUDA setup, validation, usage patterns - **NO EXPERIMENT CODE CHANGES**

## 🎯 **Mission Scope**

### **Primary Objective**
Fix CUDA compatibility to enable larger networks and longer training runs for detecting small effect sizes (~0.2-1%) that require more statistical power.

### **Deliverables**
1. ✅ **Working CUDA Setup**: GPU acceleration functional for all SPECTRA components
2. ✅ **Validation Script**: Confirm CPU vs GPU results match exactly
3. ✅ **Usage Documentation**: Clear patterns for running CUDA experiments
4. ✅ **Troubleshooting Guide**: Common issues and solutions

### **Explicit Non-Scope**
- ❌ **No experiment modifications**: Don't change configs/, scripts/, or experiment logic
- ❌ **No new features**: Focus purely on CUDA compatibility
- ❌ **No performance optimization**: Just make it work correctly first

---

## 🚨 **Current CUDA Issues**

### **Known Problem**: PyTorch CUDA Version Mismatch
```
UserWarning: Found GPU0 NVIDIA GeForce GTX 1070 which is of cuda capability 6.1.
Minimum and Maximum cuda capability supported by this version of PyTorch is (7.0) - (12.0)
```

### **Impact on Research**
- **Limited to CPU**: Small networks, short training (current 8x8, 100 epochs max)
- **Statistical Power**: Need larger networks + more epochs to detect ~0.2% effects reliably
- **Research Bottleneck**: Cannot test conditions where spectral regularization might show clearer benefits

### **Root Cause Analysis**
- **Hardware**: NVIDIA GTX 1070 (CUDA capability 6.1)
- **PyTorch Version**: Requires CUDA 7.0+ (incompatible with 6.1)
- **Solution**: Downgrade PyTorch or upgrade hardware/drivers

---

## 🔧 **Technical Context**

### **Current Environment**
```bash
# Check current setup
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# System info
nvidia-smi
lspci | grep -i nvidia
```

### **SPECTRA Components Using PyTorch**
1. **Core Models**: `spectra/models/mlp.py` - SpectralMLP class
2. **Regularizers**: `spectra/regularization/` - All spectral analysis components
3. **Training**: `spectra/training/experiment.py` - Main experiment runner
4. **Metrics**: `spectra/metrics/criticality.py` - GPU-accelerated computations

### **Device Management Pattern**
```python
# Current pattern throughout codebase
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)
```

---

## 📋 **Validation Requirements**

### **Critical Test**: CPU vs GPU Result Matching
Create validation script that confirms:
1. **Exact Numerical Results**: Same random seeds → identical outputs
2. **All Components Work**: Models, regularizers, metrics, training loop
3. **Memory Management**: No GPU memory leaks during long runs
4. **Error Handling**: Graceful fallback to CPU if GPU fails

### **Test Configuration**
```yaml
# Use existing config for validation
configs/phase3a_optimal_beta_8x8.yaml
# But limit to 3 seeds and 10 epochs for quick validation
```

### **Success Criteria**
```python
# CPU and GPU results must match to at least 6 decimal places
cpu_accuracy = 0.904123
gpu_accuracy = 0.904123  # Must match exactly
assert abs(cpu_accuracy - gpu_accuracy) < 1e-6
```

---

## 🛠 **Implementation Strategy**

### **Recommended Approach**
1. **Environment Analysis**: Determine best PyTorch version for GTX 1070
2. **Clean Installation**: Fresh virtual environment with compatible versions
3. **Component Testing**: Validate each SPECTRA module individually
4. **Integration Testing**: Full experiment run with CPU/GPU comparison
5. **Documentation**: Usage patterns and troubleshooting guide

### **PyTorch Version Strategy**
**Option 1**: Downgrade to PyTorch 1.x that supports CUDA 6.1
**Option 2**: Use CPU-only PyTorch but optimize for larger networks
**Option 3**: Docker container with compatible CUDA runtime

### **Validation Script Template**
```python
#!/usr/bin/env python3
"""
CUDA Validation Script
Confirms identical results between CPU and GPU execution
"""

def validate_cuda_setup():
    """Test all SPECTRA components on CPU vs GPU"""
    # Test 1: Model forward pass
    # Test 2: Spectral regularization computation  
    # Test 3: Training loop (single epoch)
    # Test 4: Full experiment (3 seeds, 10 epochs)
    pass

if __name__ == "__main__":
    validate_cuda_setup()
```

---

## 📚 **Resources & Context**

### **Current Working Directory Structure**
```
/home/daniel/prj/ASISR/
├── spectra/          # Main package - all components use device management
├── configs/          # Experiment configurations 
├── scripts/          # Analysis and validation scripts
├── venv/            # Current Python environment (problematic)
└── requirements.txt  # Package dependencies
```

### **Key Files to Test**
- `spectra/models/mlp.py` - Model device placement
- `spectra/regularization/adaptive.py` - Spectral analysis on GPU
- `spectra/training/experiment.py` - Training loop device management
- `run_experiment.py` - Main experiment runner

### **Current Requirements**
```bash
# From requirements.txt - may need CUDA-compatible versions
torch>=1.9.0
torchvision>=0.10.0  
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0
PyYAML>=5.4.0
```

---

## 🎯 **Success Validation**

### **Functional Tests**
1. **Basic GPU Access**: `torch.cuda.is_available() == True`
2. **Memory Allocation**: Can create and move tensors to GPU
3. **Model Training**: Single epoch runs without errors
4. **Large Networks**: Can handle 64x64+ architectures (not just 8x8)

### **Performance Baseline**
```python
# Quick performance check (don't optimize, just confirm working)
# CPU baseline: ~20s per epoch for 8x8 network
# GPU should be faster, but main goal is correctness
```

### **Integration Test**
```bash
# This should work without CUDA warnings
python run_experiment.py single configs/phase3a_optimal_beta_8x8.yaml
```

---

## 🚨 **Handover Constraints**

### **Do NOT Change**
- Any experiment configurations in `configs/`
- Experiment logic in `scripts/` or `run_experiment.py`
- Model architectures or training procedures
- Statistical validation or analysis code

### **DO Focus On**
- PyTorch/CUDA version compatibility
- Device management and memory handling
- Validation that results are identical
- Clear documentation for future users

### **Communication**
- Document any version changes in commit messages
- Note any dependencies that changed
- Provide clear setup instructions for reproducibility

---

## 📞 **Context Handoff Protocol**

**Initialization Command**:
```bash
claude --prompt "Named context: CUDA-Fix. Read docs/cuda-fix-handover.md. Fix CUDA compatibility for GTX 1070, validate identical CPU/GPU results, document usage patterns. Scope: CUDA setup only, no experiment code changes."
```

**Success Criteria**: When complete, the next context should be able to run meaningful experiments on larger networks (64x64+) and longer training runs (500+ epochs) to detect small effect sizes with proper statistical power.

**Repository Status**: All major documentation updates committed, Phase 3 lessons learned captured, ready for technical infrastructure improvement.

---

**Handover Complete**: Ready for CUDA-focused technical implementation!