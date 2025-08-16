#!/usr/bin/env python3
"""
Phase 3 Reality Check: Compare Actual Results vs Claims

Analyze the actual experimental data to understand where our interpretation went wrong
and establish the true effect sizes.
"""

import numpy as np
import scipy.stats as stats

def analyze_actual_results():
    """Compare actual experimental results with statistical rigor."""
    
    print("🔍 PHASE 3 REALITY CHECK: ACTUAL vs CLAIMED RESULTS")
    print("=" * 60)
    
    # Actual experimental results (15 seeds each)
    phase2d_linear_results = [0.893, 0.912, 0.906, 0.908, 0.906, 0.897, 0.925, 
                             0.880, 0.911, 0.915, 0.892, 0.874, 0.896, 0.897, 0.912]
    
    phase3a_adaptive_results = [0.886, 0.895, 0.901, 0.912, 0.909, 0.900, 0.928, 
                               0.878, 0.918, 0.916, 0.915, 0.889, 0.904, 0.894, 0.911]
    
    # Calculate statistics
    linear_mean = np.mean(phase2d_linear_results)
    linear_std = np.std(phase2d_linear_results, ddof=1)
    linear_se = linear_std / np.sqrt(len(phase2d_linear_results))
    
    adaptive_mean = np.mean(phase3a_adaptive_results)
    adaptive_std = np.std(phase3a_adaptive_results, ddof=1)
    adaptive_se = adaptive_std / np.sqrt(len(phase3a_adaptive_results))
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(phase3a_adaptive_results, phase2d_linear_results)
    
    print("📊 ACTUAL EXPERIMENTAL RESULTS")
    print("-" * 35)
    print(f"Phase 2D Linear (15 seeds):")
    print(f"  Mean: {linear_mean:.4f}")
    print(f"  Std:  {linear_std:.4f}")
    print(f"  95% CI: [{linear_mean - 1.96*linear_se:.4f}, {linear_mean + 1.96*linear_se:.4f}]")
    print()
    print(f"Phase 3A Adaptive (15 seeds):")
    print(f"  Mean: {adaptive_mean:.4f}")
    print(f"  Std:  {adaptive_std:.4f}")
    print(f"  95% CI: [{adaptive_mean - 1.96*adaptive_se:.4f}, {adaptive_mean + 1.96*adaptive_se:.4f}]")
    print()
    
    # Effect size analysis
    difference = adaptive_mean - linear_mean
    pooled_std = np.sqrt((linear_std**2 + adaptive_std**2) / 2)
    cohens_d = difference / pooled_std
    
    print("📈 EFFECT SIZE ANALYSIS")
    print("-" * 25)
    print(f"Raw Difference: {difference:.4f} ({difference*100:+.1f}%)")
    print(f"Effect Size (Cohen's d): {cohens_d:.4f}")
    print(f"Statistical Test: t={t_stat:.3f}, p={p_value:.4f}")
    print()
    
    # Interpret effect size
    if abs(cohens_d) < 0.2:
        effect_interp = "Negligible"
    elif abs(cohens_d) < 0.5:
        effect_interp = "Small"
    elif abs(cohens_d) < 0.8:
        effect_interp = "Medium"
    else:
        effect_interp = "Large"
    
    print("🎯 STATISTICAL INTERPRETATION")
    print("-" * 30)
    print(f"Effect Size: {effect_interp} (Cohen's d = {cohens_d:.3f})")
    if p_value < 0.05:
        print(f"Significance: SIGNIFICANT (p = {p_value:.4f})")
    else:
        print(f"Significance: NOT SIGNIFICANT (p = {p_value:.4f})")
    print()
    
    # Compare with original claims
    print("🚨 CLAIM vs REALITY COMPARISON")
    print("-" * 35)
    print("ORIGINAL CLAIMS:")
    print("  Phase 2D Baseline: 88.5% (estimated)")
    print("  Phase 3A Adaptive: 90.4%")
    print("  Claimed Improvement: +1.9% (+3.0% total swing)")
    print()
    print("ACTUAL RESULTS:")
    print(f"  Phase 2D Linear: {linear_mean:.1%}")
    print(f"  Phase 3A Adaptive: {adaptive_mean:.1%}")
    print(f"  Actual Improvement: {difference*100:+.1f}%")
    print()
    
    # Analysis of error
    estimated_baseline = 0.885
    error_in_baseline = linear_mean - estimated_baseline
    
    print("🔍 ERROR ANALYSIS")
    print("-" * 18)
    print(f"Baseline Error: {error_in_baseline*100:+.1f}% ({estimated_baseline:.1%} → {linear_mean:.1%})")
    print(f"Claimed vs Actual: {1.9 - difference*100:.1f}% overestimate")
    print()
    
    # Power analysis
    required_n_for_significance = calculate_required_sample_size(difference, pooled_std)
    print("📊 POWER ANALYSIS")
    print("-" * 17)
    print(f"Current Power: {calculate_power(difference, pooled_std, 15):.1%}")
    print(f"Required n for significance: {required_n_for_significance}")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS")
    print("-" * 18)
    if p_value > 0.05:
        print("❌ Current results do NOT support capacity-adaptive superiority")
        print("   - Effect size too small for reliable detection")
        print("   - Need ~{required_n_for_significance} seeds for 80% power")
        print("   - Consider larger architectures where effects might be clearer")
    else:
        print("✅ Results support small but significant improvement")
        print(f"   - Effect size: {effect_interp.lower()}")
        print("   - Consider practical significance vs computational cost")
    
    return {
        'linear_mean': linear_mean,
        'adaptive_mean': adaptive_mean, 
        'difference': difference,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    }

def calculate_power(effect_size, std, n, alpha=0.05):
    """Calculate statistical power for current experiment."""
    se = std * np.sqrt(2/n)  # Standard error for two-sample test
    t_critical = stats.t.ppf(1 - alpha/2, 2*n - 2)
    t_observed = effect_size / se
    power = 1 - stats.t.cdf(t_critical - t_observed, 2*n - 2)
    return power

def calculate_required_sample_size(effect_size, std, power=0.8, alpha=0.05):
    """Calculate required sample size for desired power."""
    if effect_size == 0:
        return float('inf')
    
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_beta = stats.norm.ppf(power)
    
    n = 2 * ((z_alpha + z_beta) * std / effect_size) ** 2
    return int(np.ceil(n))

def investigate_baseline_sources():
    """Investigate where the 88.5% baseline estimate came from."""
    print("\n🔍 BASELINE INVESTIGATION")
    print("=" * 30)
    
    print("POTENTIAL SOURCES OF 88.5% ESTIMATE:")
    print("1. Phase 2D reports mentioned -1.1% effect")
    print("2. May have been effect relative to no-regularization")
    print("3. Could be estimated from different architecture or dataset")
    print("4. Possible confusion between absolute accuracy and relative effect")
    print()
    
    print("NEED TO VERIFY:")
    print("- What was the actual Phase 2D no-regularization baseline?")
    print("- Was -1.1% relative to 89.6% baseline (giving 88.5%)?")
    print("- Were different experimental conditions used?")
    print("- Was this an architecture-specific effect?")

if __name__ == "__main__":
    results = analyze_actual_results()
    investigate_baseline_sources()
    
    print("\n" + "="*60)
    if results['significant']:
        print("🤏 CONCLUSION: SMALL BUT POTENTIALLY SIGNIFICANT EFFECT")
        print(f"   Capacity-adaptive shows {results['difference']*100:+.1f}% improvement")
        print("   But practical significance questionable given computational cost")
    else:
        print("❌ CONCLUSION: NO SIGNIFICANT EVIDENCE FOR BREAKTHROUGH")
        print("   Need to either increase sample size or focus on larger effects")
    print("="*60)