"""
Phase 4A: Conditions Discovery Experiment Runner

Strategic experimental design to test if spectral regularization provides >1% improvements
when scaling up from toy problems to realistic ones. Builds on existing infrastructure
with GPU acceleration and systematic parameter sweeps.
"""

import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json

import torch
import numpy as np
import pandas as pd

from .base import BaseExperiment, ExperimentResult, ComparisonResult
from ..training.experiment import SPECTRAExperiment, ExperimentResults
from ..utils.config import load_config
from ..utils.seed import set_seed


@dataclass
class Phase4AExperimentConfig:
    """Configuration for Phase 4A systematic conditions discovery."""
    
    # Architecture configurations (strategic sampling)
    architectures: List[Tuple[str, List[int]]] = None
    
    # Dataset configurations  
    datasets: List[str] = None
    
    # Training configurations
    training_configs: List[Dict[str, Any]] = None
    
    # Regularization configurations
    regularization_configs: List[Dict[str, Any]] = None
    
    # Statistical parameters
    n_seeds: int = 5
    
    # GPU/device settings
    device: str = "auto"
    
    def __post_init__(self):
        """Set default strategic configurations if not provided."""
        if self.architectures is None:
            self.architectures = [
                ("8x8", [8, 8]),      # Phase 3 baseline
                ("32x32", [32, 32]),  # 4x scale-up
                ("64x64", [64, 64]),  # 8x scale-up  
                ("128x64", [128, 64]) # Different depth/width ratio
            ]
        
        if self.datasets is None:
            self.datasets = [
                "TwoMoons",  # Known baseline
                "Circles",   # Known baseline
                "MNIST"      # Realistic scale
            ]
        
        if self.training_configs is None:
            self.training_configs = [
                {"epochs": 100, "learning_rate": 0.01, "optimizer": "adam"},  # Phase 3 baseline
                {"epochs": 500, "learning_rate": 0.01, "optimizer": "adam"}   # 5x longer
            ]
        
        if self.regularization_configs is None:
            self.regularization_configs = [
                {"type": "none"},  # Baseline
                {"type": "linear", "initial": 2.5, "final": 1.0, "strength": 0.1},  # Phase 2 validated
                {"type": "adaptive", "beta": -0.2, "strength": 0.1}  # Phase 3 implementation
            ]


class Phase4AExperiment(BaseExperiment):
    """
    Phase 4A systematic conditions discovery experiment.
    
    Tests strategic points in the experimental space to determine if spectral
    regularization benefits emerge at realistic scales.
    """
    
    def __init__(self, config: Phase4AExperimentConfig, output_base: Path = Path("plots")):
        """
        Initialize Phase 4A experiment.
        
        Args:
            config: Phase 4A experimental configuration
            output_base: Base directory for outputs
        """
        super().__init__(output_base)
        self.config = config
        self.results_db = []  # Store all experimental results
        
        # Set up device
        if torch.cuda.is_available() and config.device == "auto":
            self.device = "cuda"
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            print("💻 Using CPU")
    
    def get_phase_name(self) -> str:
        """Return phase name for output organization."""
        return "phase4a"
    
    def create_experiment_config(self, 
                                arch_name: str, 
                                arch_dims: List[int],
                                dataset: str,
                                training_config: Dict[str, Any],
                                regularization_config: Dict[str, Any],
                                seed: int) -> Dict[str, Any]:
        """
        Create a complete experiment configuration.
        
        Args:
            arch_name: Architecture name for identification
            arch_dims: Hidden layer dimensions
            dataset: Dataset name
            training_config: Training parameters
            regularization_config: Regularization parameters  
            seed: Random seed
            
        Returns:
            Complete experiment configuration dict
        """
        config = {
            "experiment": {
                "name": f"phase4a_{arch_name}_{dataset}_{regularization_config['type']}",
                "phase": "phase4a",
                "seed": seed
            },
            "device": self.device,
            "model": {
                "type": "SpectralMLP",
                "hidden_dims": arch_dims
            },
            "data": {
                "type": dataset
            },
            "training": training_config.copy(),
            "metrics": ["accuracy", "loss", "criticality_score", "spectral_radius_avg"]
        }
        
        # Add regularization if not baseline
        if regularization_config["type"] != "none":
            config["regularization"] = regularization_config.copy()
        
        # Add dataset-specific parameters
        if dataset == "TwoMoons":
            config["data"].update({
                "n_samples": 1000,
                "noise": 0.1,
                "random_state": 42
            })
            config["model"]["input_dim"] = 2
            config["model"]["output_dim"] = 2
        elif dataset == "Circles":
            config["data"].update({
                "n_samples": 1000,
                "noise": 0.1,
                "factor": 0.5,
                "random_state": 42
            })
            config["model"]["input_dim"] = 2
            config["model"]["output_dim"] = 2
        elif dataset == "MNIST":
            config["data"].update({
                "flatten": True,  # For MLP compatibility
                "normalize": True
            })
            config["model"]["input_dim"] = 784  # 28*28 flattened
            config["model"]["output_dim"] = 10  # 10 digit classes
        
        return config
    
    def run_single_condition(self,
                           arch_name: str,
                           arch_dims: List[int], 
                           dataset: str,
                           training_config: Dict[str, Any],
                           regularization_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single experimental condition across multiple seeds.
        
        Args:
            arch_name: Architecture name
            arch_dims: Hidden layer dimensions
            dataset: Dataset name
            training_config: Training parameters
            regularization_config: Regularization parameters
            
        Returns:
            Aggregated results across seeds
        """
        condition_name = f"{arch_name}_{dataset}_{regularization_config['type']}"
        print(f"\n{'='*60}")
        print(f"Running Phase 4A Condition: {condition_name}")
        print(f"Architecture: {arch_dims}, Dataset: {dataset}")
        print(f"Training: {training_config['epochs']} epochs")
        print(f"Regularization: {regularization_config['type']}")
        print(f"{'='*60}")
        
        seed_results = []
        
        for seed_idx, seed in enumerate(range(self.config.n_seeds)):
            print(f"  Seed {seed_idx+1}/{self.config.n_seeds} (seed={seed})")
            
            # Create config for this seed
            exp_config = self.create_experiment_config(
                arch_name, arch_dims, dataset, training_config, regularization_config, seed
            )
            
            try:
                # Run experiment
                set_seed(seed)
                
                # Create temporary config file
                import tempfile
                import yaml
                temp_config_path = None
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                    yaml.dump(exp_config, f)
                    temp_config_path = f.name
                
                from ..utils.config import SPECTRAConfig
                spectra_config = SPECTRAConfig(temp_config_path)
                experiment = SPECTRAExperiment(spectra_config)
                start_time = time.time()
                results = experiment.run_single_seed(seed)
                execution_time = time.time() - start_time
                
                # Clean up temporary file
                if temp_config_path:
                    Path(temp_config_path).unlink()
                
                # Extract key metrics (handle both dict and object results)
                if hasattr(results, 'metrics_history'):
                    metrics_history = results.metrics_history
                elif isinstance(results, dict) and 'metrics_history' in results:
                    metrics_history = results['metrics_history']
                else:
                    print(f"    Debug: results type = {type(results)}")
                    print(f"    Debug: results keys = {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
                    metrics_history = {}
                
                seed_result = {
                    "seed": seed,
                    "execution_time": execution_time,
                    "final_accuracy": metrics_history.get("accuracy", [0.0])[-1] if metrics_history.get("accuracy") else 0.0,
                    "final_loss": metrics_history.get("loss", [float('inf')])[-1] if metrics_history.get("loss") else float('inf'),
                    "final_criticality": metrics_history.get("criticality_score", [0.0])[-1] if metrics_history.get("criticality_score") else 0.0
                }
                
                seed_results.append(seed_result)
                print(f"    Accuracy: {seed_result['final_accuracy']:.3f}, Time: {execution_time:.1f}s")
                
            except Exception as e:
                print(f"    ❌ Failed: {e}")
                # Add failed result
                seed_results.append({
                    "seed": seed,
                    "execution_time": 0,
                    "final_accuracy": 0.0,
                    "final_loss": float('inf'),
                    "final_criticality": 0.0,
                    "error": str(e)
                })
        
        # Aggregate results
        valid_results = [r for r in seed_results if "error" not in r]
        
        if valid_results:
            accuracies = [r["final_accuracy"] for r in valid_results]
            condition_result = {
                "condition_name": condition_name,
                "architecture": arch_name,
                "dataset": dataset,
                "training_config": training_config,
                "regularization_config": regularization_config,
                "n_valid_seeds": len(valid_results),
                "accuracy_mean": np.mean(accuracies),
                "accuracy_std": np.std(accuracies, ddof=1) if len(accuracies) > 1 else 0.0,
                "accuracy_min": np.min(accuracies),
                "accuracy_max": np.max(accuracies),
                "total_execution_time": sum(r["execution_time"] for r in valid_results),
                "seed_results": seed_results
            }
        else:
            condition_result = {
                "condition_name": condition_name,
                "architecture": arch_name,
                "dataset": dataset,
                "training_config": training_config,
                "regularization_config": regularization_config,
                "n_valid_seeds": 0,
                "accuracy_mean": 0.0,
                "accuracy_std": 0.0,
                "accuracy_min": 0.0,
                "accuracy_max": 0.0,
                "total_execution_time": 0,
                "seed_results": seed_results,
                "error": "All seeds failed"
            }
        
        # Store in results database
        self.results_db.append(condition_result)
        
        print(f"  ✅ Condition complete: {condition_result['accuracy_mean']:.3f} ± {condition_result['accuracy_std']:.3f}")
        
        return condition_result
    
    def run_baseline_experiments(self) -> List[Dict[str, Any]]:
        """
        Run baseline experiments (no regularization) to establish benchmarks.
        
        Returns:
            List of baseline experimental results
        """
        print(f"\n🎯 Running Phase 4A Baseline Experiments")
        print(f"Establishing performance benchmarks across conditions")
        
        baseline_results = []
        baseline_reg_config = {"type": "none"}
        
        # Run baselines for all configured conditions
        for arch_name, arch_dims in self.config.architectures:
            for dataset in self.config.datasets:
                for training_config in self.config.training_configs:
                    result = self.run_single_condition(
                        arch_name, arch_dims, dataset, training_config, baseline_reg_config
                    )
                    baseline_results.append(result)
        
        return baseline_results
    
    def run_spectral_experiments(self, baseline_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run spectral regularization experiments on same conditions as baselines.
        
        Args:
            baseline_results: Baseline results for comparison
            
        Returns:
            List of spectral regularization experimental results
        """
        print(f"\n🔬 Running Phase 4A Spectral Regularization Experiments")
        
        spectral_results = []
        
        # Extract tested conditions from baselines
        tested_conditions = set()
        for baseline in baseline_results:
            if "error" not in baseline:
                condition_key = (
                    baseline["architecture"], 
                    tuple(baseline["regularization_config"]['type'] == 'none'),  # placeholder
                    baseline["dataset"],
                    baseline["training_config"]["epochs"]
                )
                tested_conditions.add((
                    baseline["architecture"],
                    baseline["dataset"], 
                    baseline["training_config"]
                ))
        
        # Run spectral regularization on same conditions
        spectral_reg_configs = [
            reg_config for reg_config in self.config.regularization_configs 
            if reg_config["type"] != "none"
        ]
        
        for arch_name, dataset, training_config in tested_conditions:
            # Get architecture dimensions
            arch_dims = dict(self.config.architectures)[arch_name]
            
            for reg_config in spectral_reg_configs:
                result = self.run_single_condition(
                    arch_name, arch_dims, dataset, training_config, reg_config
                )
                spectral_results.append(result)
        
        return spectral_results
    
    def analyze_results(self, baseline_results: List[Dict[str, Any]], 
                       spectral_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze experimental results and compute effect sizes.
        
        Args:
            baseline_results: Baseline experimental results
            spectral_results: Spectral regularization results
            
        Returns:
            Analysis summary with effect sizes and recommendations
        """
        print(f"\n📊 Analyzing Phase 4A Results")
        
        analysis = {
            "total_experiments": len(baseline_results) + len(spectral_results),
            "successful_experiments": len([r for r in baseline_results + spectral_results if "error" not in r]),
            "comparisons": [],
            "summary_statistics": {},
            "recommendations": {}
        }
        
        # Create baseline lookup
        baseline_lookup = {}
        for baseline in baseline_results:
            if "error" not in baseline:
                key = (baseline["architecture"], baseline["dataset"], baseline["training_config"]["epochs"])
                baseline_lookup[key] = baseline
        
        # Compare spectral results to matched baselines
        significant_improvements = []
        
        for spectral in spectral_results:
            if "error" in spectral:
                continue
                
            key = (spectral["architecture"], spectral["dataset"], spectral["training_config"]["epochs"])
            
            if key in baseline_lookup:
                baseline = baseline_lookup[key]
                
                # Calculate effect size
                baseline_acc = baseline["accuracy_mean"]
                spectral_acc = spectral["accuracy_mean"]
                improvement = spectral_acc - baseline_acc
                relative_improvement = improvement / baseline_acc if baseline_acc > 0 else 0
                
                # Simple significance test (proper statistical testing would require raw data)
                baseline_std = baseline["accuracy_std"]
                spectral_std = spectral["accuracy_std"]
                pooled_std = np.sqrt((baseline_std**2 + spectral_std**2) / 2)
                effect_size = improvement / pooled_std if pooled_std > 0 else 0
                
                comparison = {
                    "condition": f"{spectral['architecture']}_{spectral['dataset']}_{spectral['training_config']['epochs']}ep",
                    "regularization_type": spectral["regularization_config"]["type"],
                    "baseline_accuracy": baseline_acc,
                    "spectral_accuracy": spectral_acc,
                    "improvement_absolute": improvement,
                    "improvement_relative": relative_improvement,
                    "effect_size": effect_size,
                    "potentially_significant": abs(effect_size) > 0.5,  # Rough threshold
                    "practically_meaningful": abs(relative_improvement) > 0.01  # >1% improvement
                }
                
                analysis["comparisons"].append(comparison)
                
                if comparison["practically_meaningful"] and comparison["potentially_significant"]:
                    significant_improvements.append(comparison)
        
        # Summary statistics
        if analysis["comparisons"]:
            improvements = [c["improvement_relative"] for c in analysis["comparisons"]]
            analysis["summary_statistics"] = {
                "mean_relative_improvement": np.mean(improvements),
                "std_relative_improvement": np.std(improvements),
                "max_improvement": np.max(improvements),
                "min_improvement": np.min(improvements),
                "n_positive_effects": sum(1 for imp in improvements if imp > 0),
                "n_meaningful_effects": len(significant_improvements),
                "fraction_positive": sum(1 for imp in improvements if imp > 0) / len(improvements)
            }
        
        # Generate recommendations
        n_meaningful = len(significant_improvements)
        
        if n_meaningful >= 2:
            analysis["recommendations"]["decision"] = "CONTINUE"
            analysis["recommendations"]["rationale"] = f"Found {n_meaningful} conditions with >1% improvements"
            analysis["recommendations"]["next_steps"] = "Expand to full systematic sweep"
        elif n_meaningful == 1:
            analysis["recommendations"]["decision"] = "INVESTIGATE" 
            analysis["recommendations"]["rationale"] = "Found 1 promising condition - needs validation"
            analysis["recommendations"]["next_steps"] = "Focus on promising condition with more seeds"
        else:
            analysis["recommendations"]["decision"] = "PIVOT"
            analysis["recommendations"]["rationale"] = "No meaningful improvements found in strategic sample"
            analysis["recommendations"]["next_steps"] = "Consider alternative research directions"
        
        return analysis
    
    def save_results(self, analysis: Dict[str, Any]) -> Path:
        """
        Save comprehensive results to JSON file.
        
        Args:
            analysis: Analysis results to save
            
        Returns:
            Path to saved results file
        """
        output_dir = self.get_output_dir()
        results_file = output_dir / "phase4a_results.json"
        
        # Prepare data for JSON serialization
        save_data = {
            "config": {
                "architectures": self.config.architectures,
                "datasets": self.config.datasets,
                "training_configs": self.config.training_configs,
                "regularization_configs": self.config.regularization_configs,
                "n_seeds": self.config.n_seeds,
                "device": self.device
            },
            "results_db": self.results_db,
            "analysis": analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {results_file}")
        return results_file
    
    def run(self, run_baseline: bool = True, run_spectral: bool = True) -> Dict[str, Any]:
        """
        Run Phase 4A conditions discovery experiment.
        
        Args:
            run_baseline: Whether to run baseline experiments
            run_spectral: Whether to run spectral regularization experiments
            
        Returns:
            Complete experimental analysis
        """
        print(f"\n🚀 Starting Phase 4A: Systematic Conditions Discovery")
        print(f"Device: {self.device}")
        print(f"Total planned experiments: {len(self.config.architectures) * len(self.config.datasets) * len(self.config.training_configs) * len(self.config.regularization_configs) * self.config.n_seeds}")
        
        start_time = time.time()
        
        # Run baseline experiments
        baseline_results = []
        if run_baseline:
            baseline_results = self.run_baseline_experiments()
        
        # Run spectral regularization experiments  
        spectral_results = []
        if run_spectral and baseline_results:
            spectral_results = self.run_spectral_experiments(baseline_results)
        
        # Analyze results
        analysis = self.analyze_results(baseline_results, spectral_results)
        analysis["total_execution_time"] = time.time() - start_time
        
        # Save results
        self.save_results(analysis)
        
        # Print summary
        self.print_summary(analysis)
        
        return analysis
    
    def print_summary(self, analysis: Dict[str, Any]) -> None:
        """Print experiment summary to console."""
        print(f"\n{'='*80}")
        print(f"PHASE 4A RESULTS SUMMARY")
        print(f"{'='*80}")
        
        print(f"Total experiments: {analysis['total_experiments']}")
        print(f"Successful experiments: {analysis['successful_experiments']}")
        print(f"Execution time: {analysis['total_execution_time']:.1f} seconds")
        
        if analysis['summary_statistics']:
            stats = analysis['summary_statistics']
            print(f"\nEffect Size Summary:")
            print(f"  Mean relative improvement: {stats['mean_relative_improvement']:+.1%}")
            print(f"  Best improvement: {stats['max_improvement']:+.1%}")
            print(f"  Worst effect: {stats['min_improvement']:+.1%}")
            print(f"  Conditions with positive effects: {stats['n_positive_effects']}/{len(analysis['comparisons'])}")
            print(f"  Conditions with >1% improvement: {stats['n_meaningful_effects']}")
        
        print(f"\nRecommendation: {analysis['recommendations']['decision']}")
        print(f"Rationale: {analysis['recommendations']['rationale']}")
        print(f"Next steps: {analysis['recommendations']['next_steps']}")
        
        if analysis['comparisons']:
            print(f"\nTop 3 Results:")
            sorted_comparisons = sorted(
                analysis['comparisons'], 
                key=lambda x: x['improvement_relative'], 
                reverse=True
            )[:3]
            
            for i, comp in enumerate(sorted_comparisons, 1):
                print(f"  {i}. {comp['condition']} ({comp['regularization_type']}): {comp['improvement_relative']:+.1%}")
        
        print(f"{'='*80}")
    
    def generate_plots(self, result) -> List[Path]:
        """Generate visualization plots for Phase 4A results."""
        # TODO: Implement visualization when needed
        return []