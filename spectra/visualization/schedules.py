"""
σ Schedule Visualization Suite for SPECTRA Phase 2C.

Interactive visualization and exploration of dynamic spectral regularization schedules.
Shows what different scheduling algorithms actually do and enables parameter exploration.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from ..regularization.dynamic import (
    LinearScheduleRegularizer,
    ExponentialScheduleRegularizer,
    StepScheduleRegularizer,
    AdaptiveScheduleRegularizer
)


class SigmaScheduleVisualizer:
    """
    Interactive visualization suite for σ scheduling strategies.
    
    Shows σ(t) evolution curves, enables parameter exploration,
    and reveals the mechanism behind dynamic spectral control.
    """
    
    def __init__(self):
        """Initialize σ schedule visualizer with publication styling."""
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def plot_sigma_schedules(self, 
                           strategies: Dict[str, Dict[str, Any]],
                           epochs: int = 100,
                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Plot σ(t) evolution curves for different scheduling strategies.
        
        Args:
            strategies: Dict mapping strategy names to parameter dicts
            epochs: Number of training epochs to simulate
            figsize: Figure dimensions
            
        Returns:
            Matplotlib figure with σ schedule comparisons
            
        Example:
            strategies = {
                'Linear': {'initial_sigma': 2.5, 'final_sigma': 1.0},
                'Exponential': {'initial_sigma': 2.5, 'final_sigma': 1.0, 'decay_rate': 5.0},
                'Step': {'sigma_schedule': [(0, 2.5), (33, 1.7), (67, 1.0)]}
            }
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('SPECTRA Dynamic σ Scheduling Strategies', fontsize=16, fontweight='bold')
        
        epoch_range = np.arange(epochs)
        
        # Left panel: σ(t) evolution curves
        for strategy_name, params in strategies.items():
            sigma_curve = self._compute_sigma_curve(strategy_name, params, epochs)
            ax1.plot(epoch_range, sigma_curve, linewidth=3, label=strategy_name, marker='o', markersize=2)
        
        ax1.set_title('σ(t) Evolution During Training', fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('σ Target Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0.8, 2.6)
        
        # Right panel: Schedule comparison table
        ax2.axis('off')
        table_data = []
        headers = ['Strategy', 'Start σ', 'End σ', 'Pattern', 'Use Case']
        
        for strategy_name, params in strategies.items():
            sigma_curve = self._compute_sigma_curve(strategy_name, params, epochs)
            start_sigma = f"{sigma_curve[0]:.2f}"
            end_sigma = f"{sigma_curve[-1]:.2f}"
            
            if strategy_name == 'Linear':
                pattern = "Smooth decline"
                use_case = "Balanced exploration→exploitation"
            elif strategy_name == 'Exponential':
                pattern = "Fast early decay"
                use_case = "Quick convergence focus"
            elif strategy_name == 'Step':
                pattern = "Discrete phases"
                use_case = "Phase-based training"
            else:
                pattern = "Adaptive"
                use_case = "Data-driven optimization"
            
            table_data.append([strategy_name, start_sigma, end_sigma, pattern, use_case])
        
        table = ax2.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0.2, 1, 0.6])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2)
        
        # Style table
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#E8E8E8')
            table[(0, i)].set_text_props(weight='bold')
        
        ax2.set_title('Strategy Characteristics', fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
    
    def plot_sigma_with_performance(self,
                                   experiment_results: Dict[str, Any],
                                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot σ evolution with performance overlay - the key mechanism visualization.
        
        Args:
            experiment_results: Results from Phase 2B experiments with trajectory data
            figsize: Figure dimensions
            
        Returns:
            Figure showing σ(t) and accuracy(t) correlation
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('SPECTRA Mechanism: σ Scheduling → Performance Evolution', fontsize=16, fontweight='bold')
        
        # Top row: σ evolution and accuracy overlay
        ax_sigma = axes[0, 0]
        ax_acc = axes[0, 1]
        ax_combined = axes[1, 0]
        ax_phase = axes[1, 1]
        
        for strategy_name, results in experiment_results.items():
            if 'aggregated_results' not in results:
                continue
                
            agg_results = results['aggregated_results']
            
            # Extract trajectory data
            if 'accuracy' in agg_results and 'trajectory_mean' in agg_results['accuracy']:
                epochs = np.arange(len(agg_results['accuracy']['trajectory_mean']))
                accuracy_mean = agg_results['accuracy']['trajectory_mean']
                accuracy_std = agg_results['accuracy']['trajectory_std']
                
                # Plot accuracy evolution
                ax_acc.plot(epochs, accuracy_mean, linewidth=2, label=strategy_name)
                ax_acc.fill_between(epochs, 
                                  accuracy_mean - accuracy_std,
                                  accuracy_mean + accuracy_std,
                                  alpha=0.2)
                
            # Extract σ trajectory if available
            if 'sigma_target' in agg_results and 'trajectory_mean' in agg_results['sigma_target']:
                sigma_mean = agg_results['sigma_target']['trajectory_mean']
                
                # Plot σ evolution
                ax_sigma.plot(epochs, sigma_mean, linewidth=2, label=strategy_name, linestyle='--')
                
                # Combined plot: dual y-axis
                ax_combined_twin = ax_combined.twinx()
                line1 = ax_combined.plot(epochs, accuracy_mean, linewidth=2, label=f'{strategy_name} Accuracy')
                line2 = ax_combined_twin.plot(epochs, sigma_mean, linewidth=2, linestyle='--', 
                                            label=f'{strategy_name} σ Target', alpha=0.7)
                
                # Phase analysis
                if strategy_name == 'Linear':
                    # Show exploration vs exploitation phases
                    exploration_phase = epochs[sigma_mean > 1.8]
                    exploitation_phase = epochs[sigma_mean < 1.3]
                    
                    if len(exploration_phase) > 0 and len(exploitation_phase) > 0:
                        ax_phase.axvspan(exploration_phase[0], exploration_phase[-1], 
                                       alpha=0.3, color='red', label='Exploration (high σ)')
                        ax_phase.axvspan(exploitation_phase[0], exploitation_phase[-1], 
                                       alpha=0.3, color='blue', label='Exploitation (low σ)')
                        ax_phase.plot(epochs, accuracy_mean, linewidth=2, color='black', 
                                    label=f'{strategy_name} Accuracy')
        
        # Styling
        ax_sigma.set_title('σ Target Evolution')
        ax_sigma.set_ylabel('σ Value')
        ax_sigma.legend()
        ax_sigma.grid(True, alpha=0.3)
        
        ax_acc.set_title('Accuracy Evolution')  
        ax_acc.set_ylabel('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True, alpha=0.3)
        
        ax_combined.set_title('σ Schedule → Performance Mechanism')
        ax_combined.set_xlabel('Epoch')
        ax_combined.set_ylabel('Accuracy', color='tab:blue')
        ax_combined_twin.set_ylabel('σ Target', color='tab:orange')
        ax_combined.grid(True, alpha=0.3)
        
        ax_phase.set_title('Training Phase Analysis')
        ax_phase.set_xlabel('Epoch') 
        ax_phase.set_ylabel('Accuracy')
        ax_phase.legend()
        ax_phase.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def create_parameter_sensitivity_surface(self,
                                           strategy_type: str = 'linear',
                                           param_ranges: Dict[str, np.ndarray] = None,
                                           epochs: int = 100,
                                           figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create parameter sensitivity surface for schedule exploration.
        
        Args:
            strategy_type: Type of scheduling strategy to analyze
            param_ranges: Dict of parameter ranges to explore
            epochs: Training epochs to simulate
            figsize: Figure dimensions
            
        Returns:
            Surface plot showing parameter sensitivity
        """
        if param_ranges is None:
            if strategy_type == 'linear':
                param_ranges = {
                    'initial_sigma': np.linspace(1.5, 3.0, 15),
                    'final_sigma': np.linspace(0.8, 1.5, 15)
                }
            elif strategy_type == 'exponential':
                param_ranges = {
                    'decay_rate': np.linspace(1.0, 10.0, 15),
                    'final_sigma': np.linspace(0.8, 1.5, 15)
                }
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        param_names = list(param_ranges.keys())
        param1_range = param_ranges[param_names[0]]
        param2_range = param_ranges[param_names[1]]
        
        # Create parameter mesh
        P1, P2 = np.meshgrid(param1_range, param2_range)
        
        # Compute schedule characteristics for each parameter combination
        schedule_variance = np.zeros_like(P1)
        final_values = np.zeros_like(P1)
        
        for i, p1 in enumerate(param1_range):
            for j, p2 in enumerate(param2_range):
                if strategy_type == 'linear':
                    params = {'initial_sigma': p1, 'final_sigma': p2}
                elif strategy_type == 'exponential':
                    params = {'initial_sigma': 2.5, 'decay_rate': p1, 'final_sigma': p2}
                
                sigma_curve = self._compute_sigma_curve(strategy_type, params, epochs)
                schedule_variance[j, i] = np.var(sigma_curve)
                final_values[j, i] = sigma_curve[-1]
        
        # Plot surface
        surf = ax.plot_surface(P1, P2, schedule_variance, cmap='viridis', alpha=0.7)
        
        ax.set_xlabel(param_names[0].replace('_', ' ').title())
        ax.set_ylabel(param_names[1].replace('_', ' ').title())
        ax.set_zlabel('Schedule Variance')
        ax.set_title(f'{strategy_type.title()} Schedule Parameter Sensitivity')
        
        fig.colorbar(surf)
        
        return fig
    
    def _compute_sigma_curve(self, strategy_type: str, params: Dict[str, Any], epochs: int) -> np.ndarray:
        """Compute σ values over training epochs for given strategy and parameters."""
        if strategy_type.lower() == 'linear':
            regularizer = LinearScheduleRegularizer(
                initial_sigma=params.get('initial_sigma', 2.5),
                final_sigma=params.get('final_sigma', 1.0),
                total_epochs=epochs,
                regularization_strength=0.1
            )
        elif strategy_type.lower() == 'exponential':
            regularizer = ExponentialScheduleRegularizer(
                initial_sigma=params.get('initial_sigma', 2.5),
                final_sigma=params.get('final_sigma', 1.0),
                decay_rate=params.get('decay_rate', 5.0),
                total_epochs=epochs,
                regularization_strength=0.1
            )
        elif strategy_type.lower() == 'step':
            regularizer = StepScheduleRegularizer(
                sigma_schedule=params.get('sigma_schedule', [(0, 2.5), (33, 1.7), (67, 1.0)]),
                total_epochs=epochs,
                regularization_strength=0.1
            )
        else:
            # Fallback to linear
            regularizer = LinearScheduleRegularizer(2.5, 1.0, epochs, 0.1)
        
        # Simulate σ evolution
        sigma_curve = []
        for epoch in range(epochs):
            regularizer.update_epoch(epoch)
            sigma_curve.append(regularizer.current_sigma)
        
        return np.array(sigma_curve)


def save_schedule_gallery(output_dir: str = "plots/phase2c") -> None:
    """
    Generate and save complete σ schedule visualization gallery.
    
    Args:
        output_dir: Directory to save visualization plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    visualizer = SigmaScheduleVisualizer()
    
    # Standard strategy comparison
    strategies = {
        'Linear': {'initial_sigma': 2.5, 'final_sigma': 1.0},
        'Exponential': {'initial_sigma': 2.5, 'final_sigma': 1.0, 'decay_rate': 5.0},
        'Step': {'sigma_schedule': [(0, 2.5), (33, 1.7), (67, 1.0)]}
    }
    
    fig1 = visualizer.plot_sigma_schedules(strategies)
    fig1.savefig(output_path / 'sigma_schedule_comparison.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # Parameter sensitivity surfaces
    fig2 = visualizer.create_parameter_sensitivity_surface('linear')
    fig2.savefig(output_path / 'linear_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    
    fig3 = visualizer.create_parameter_sensitivity_surface('exponential')
    fig3.savefig(output_path / 'exponential_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig3)
    
    print(f"σ schedule gallery saved to {output_path}")


if __name__ == "__main__":
    # Generate visualization gallery
    save_schedule_gallery()