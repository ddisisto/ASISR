#!/bin/bash
# SPECTRA: Regenerate all plots with standardized output paths

set -e  # Exit on any error

echo "🚀 SPECTRA: Regenerating all plots with standardized paths"
echo "================================================================"

# Phase 2B: Comprehensive dynamic vs static comparison
echo "📊 Phase 2B: Dynamic vs Static Spectral Control Comparison"
python run_experiment.py phase2b \
  --static configs/phase2b_static_comparison.yaml \
  --dynamic configs/phase2b_linear_schedule.yaml \
        configs/phase2b_exponential_schedule.yaml \
        configs/phase2b_step_schedule.yaml \
  --names Linear Exponential Step \
  --plots

# Phase 2C: σ scheduling visualization suite  
echo ""
echo "📈 Phase 2C: σ Scheduling Visualization Suite"
python -c "
from spectra.visualization.schedules import save_schedule_gallery
save_schedule_gallery('plots/phase2c')
print('Phase 2C visualization gallery saved to plots/phase2c/')
"

echo ""
echo "✅ All plots regenerated successfully!"
echo "📁 Standard output locations:"
echo "   - Phase 2B: plots/phase2b/"
echo "   - Phase 2C: plots/phase2c/"
echo "================================================================"