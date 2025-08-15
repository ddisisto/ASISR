#!/bin/bash
echo "🚀 SPECTRA Comprehensive CLI Validation (Robust)"
echo "==============================================="

# Capture all output and continue on errors
LOGFILE="validation_results.log"
echo "📝 Logging all output to: $LOGFILE"
echo "" > $LOGFILE

# Function to run command and log results
run_test() {
    local test_name="$1"
    shift
    echo ""
    echo "🧪 Testing: $test_name"
    echo "🧪 Testing: $test_name" >> $LOGFILE
    echo "Command: $*" >> $LOGFILE
    echo "----------------------------------------" >> $LOGFILE
    
    if "$@" >> $LOGFILE 2>&1; then
        echo "✅ SUCCESS: $test_name"
        echo "✅ SUCCESS: $test_name" >> $LOGFILE
    else
        echo "❌ FAILED: $test_name (exit code: $?)"
        echo "❌ FAILED: $test_name (exit code: $?)" >> $LOGFILE
    fi
    echo "----------------------------------------" >> $LOGFILE
    echo "" >> $LOGFILE
}

# Phase 2B comprehensive comparison
run_test "Phase 2B Dynamic vs Static" \
    python run_experiment.py phase2b \
    --static configs/phase2b_static_comparison.yaml \
    --dynamic configs/phase2b_linear_schedule.yaml \
             configs/phase2b_exponential_schedule.yaml \
             configs/phase2b_step_schedule.yaml \
    --names Linear Exponential Step \
    --plots

# Phase 1 comparison mode
run_test "Phase 1 Comparison Mode" \
    python run_experiment.py phase1 configs/phase1_baseline.yaml --comparison

# Phase 1 single experiment  
run_test "Phase 1 Single Experiment" \
    python run_experiment.py single configs/phase1_spectral.yaml

# Additional visualization
run_test "Phase 2C Visualization (additional)" \
    python run_experiment.py visualize --output-dir plots/phase2c_validation

echo ""
echo "📊 VALIDATION SUMMARY:"
echo "====================="
grep "SUCCESS\|FAILED" $LOGFILE | grep -v "Testing:"
echo ""
echo "📁 Generated outputs:"
find plots/ -name "*.png" | wc -l | xargs echo "  Plot files generated:"
ls plots/ | xargs echo "  Output directories:"
echo ""
echo "📝 Full log available at: $LOGFILE"
echo "====================="
