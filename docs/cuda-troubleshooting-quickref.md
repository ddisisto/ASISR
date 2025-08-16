# CUDA Troubleshooting Quick Reference

**For SPECTRA Project - GTX 1070 Setup**

---

## 🔍 **Quick Diagnostics**

```bash
# 1. Check CUDA availability
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# 2. Verify GPU hardware
nvidia-smi

# 3. Check PyTorch version
python -c "import torch; print(torch.__version__)"

# 4. Run validation suite
python scripts/validate_cuda_setup.py
```

---

## ⚡ **Common Fixes**

### **GPU Not Detected**
```bash
# Check driver
nvidia-smi
# If not working, reinstall NVIDIA drivers

# Check PyTorch installation
pip show torch
# Should show: torch 2.7.1+cu118 or compatible
```

### **Wrong PyTorch Version**
```bash
# Uninstall and reinstall correct version
pip uninstall torch torchvision torchaudio -y
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
```

### **Kernel Compatibility Error**
```
Error: CUDA error: no kernel image is available for execution on the device
```
**Fix**: Install PyTorch with CUDA 11.8 (see above command)

### **Memory Errors**
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check memory usage
python -c "import torch; print(f'Memory: {torch.cuda.memory_allocated()/1024**2:.1f}MB')"
```

### **Experiments Still Run on CPU**
```yaml
# Check config file for:
device: "cpu"

# Change to:
device: "cuda"  # or "auto"
```

---

## 📊 **Expected Outputs**

### **Working CUDA Setup**
```
CUDA: True
PyTorch: 2.7.1+cu118
Device: cuda
All tests passed: ✅
```

### **Experiment Output Should Show**
```
Device: cuda
Completed in X.Xs (faster than CPU)
```

---

## 🚨 **When to Get Help**

**Stop and investigate if:**
- Validation script shows multiple failures
- GPU performance slower than CPU
- Memory usage >1GB for small models
- Frequent CUDA errors during training

**Normal warnings (ignore):**
- CUBLAS_WORKSPACE_CONFIG deterministic behavior warnings
- GTX 1070 capability 6.1 compatibility notices

---

## 📞 **Emergency Fallback**

```bash
# Restore original environment
pip uninstall torch torchvision torchaudio -y
pip install -r requirements.txt.backup

# Use CPU mode
# Change all configs: device: "cpu"
```

---

**Quick Help**: All experiments can run on CPU if GPU fails - just slower.