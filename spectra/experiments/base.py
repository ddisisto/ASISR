"""
Base experiment classes for standardized SPECTRA research.

Provides common interfaces and patterns for all experiment types,
ensuring consistent output formats and reproducible execution.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import time
from dataclasses import dataclass

from ..training.experiment import SPECTRAExperiment, ExperimentResults
from ..utils.config import load_config


@dataclass
class ExperimentResult:
    """Standardized result container for single experiments."""
    experiment_name: str
    config_path: str
    results: ExperimentResults
    execution_time: float
    output_dir: Optional[Path] = None
    
    @property
    def final_accuracy(self) -> float:
        """Get final accuracy mean."""
        if 'accuracy' in self.results.aggregated_results:
            return self.results.aggregated_results['accuracy']['final_mean']
        return 0.0
    
    @property  
    def final_accuracy_std(self) -> float:
        """Get final accuracy standard deviation."""
        if 'accuracy' in self.results.aggregated_results:
            return self.results.aggregated_results['accuracy']['final_std']
        return 0.0


@dataclass
class ComparisonResult:
    """Standardized result container for experiment comparisons."""
    experiment_name: str
    baseline_result: ExperimentResult
    comparison_results: List[ExperimentResult]
    statistical_tests: Dict[str, Any]
    output_dir: Optional[Path] = None
    
    def get_results_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all results for easy comparison."""
        summary = {
            'baseline': {
                'accuracy_mean': self.baseline_result.final_accuracy,
                'accuracy_std': self.baseline_result.final_accuracy_std
            }
        }
        
        for result in self.comparison_results:
            summary[result.experiment_name] = {
                'accuracy_mean': result.final_accuracy,
                'accuracy_std': result.final_accuracy_std
            }
        
        return summary


class BaseExperiment(ABC):
    """
    Base class for all SPECTRA experiments.
    
    Provides standardized execution patterns, output management,
    and result formatting across all research phases.
    """
    
    def __init__(self, output_base: Path = Path("plots")):
        """
        Initialize base experiment.
        
        Args:
            output_base: Base directory for all experiment outputs
        """
        self.output_base = output_base
        self.output_base.mkdir(exist_ok=True)
    
    @abstractmethod
    def get_phase_name(self) -> str:
        """Return the research phase name (e.g., 'phase1', 'phase2b')."""
        pass
    
    @abstractmethod
    def run(self, **kwargs) -> Union[ExperimentResult, ComparisonResult]:
        """Run the experiment and return standardized results."""
        pass
    
    def get_output_dir(self, experiment_name: Optional[str] = None) -> Path:
        """
        Get standardized output directory for this experiment.
        
        Args:
            experiment_name: Optional sub-experiment name
            
        Returns:
            Path to experiment output directory
        """
        phase_dir = self.output_base / self.get_phase_name()
        phase_dir.mkdir(exist_ok=True)
        
        if experiment_name:
            exp_dir = phase_dir / experiment_name
            exp_dir.mkdir(exist_ok=True)
            return exp_dir
        
        return phase_dir
    
    def run_single_experiment(self, 
                            config_path: str,
                            experiment_name: str) -> ExperimentResult:
        """
        Run a single experiment with standardized execution pattern.
        
        Args:
            config_path: Path to YAML configuration file
            experiment_name: Name for this experiment
            
        Returns:
            Standardized experiment result
        """
        print(f"{'='*60}")
        print(f"Running {self.get_phase_name().upper()}: {experiment_name}")
        print(f"Config: {Path(config_path).stem}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Load config and run experiment
        config = load_config(config_path)
        experiment = SPECTRAExperiment(config)
        results = experiment.run_multi_seed()
        results.aggregate_results()
        
        execution_time = time.time() - start_time
        
        # Print summary
        if 'accuracy' in results.aggregated_results:
            acc_stats = results.aggregated_results['accuracy']
            print(f"{experiment_name}: {acc_stats['final_mean']:.3f} ± {acc_stats['final_std']:.3f} accuracy")
        
        return ExperimentResult(
            experiment_name=experiment_name,
            config_path=config_path,
            results=results,
            execution_time=execution_time,
            output_dir=self.get_output_dir(experiment_name)
        )
    
    @abstractmethod
    def generate_plots(self, 
                      result: Union[ExperimentResult, ComparisonResult]) -> List[Path]:
        """
        Generate standardized plots for experiment results.
        
        Args:
            result: Experiment result(s) to visualize
            
        Returns:
            List of generated plot file paths
        """
        pass