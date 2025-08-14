"""
Criticality assessment and monitoring for neural networks.

This module implements unified criticality metrics combining multiple
indicators to assess whether networks are operating at the "edge of chaos".
"""

from typing import Dict, List, Optional
import torch
import numpy as np
from ..models.base import SpectralRegularizedModel


class CriticalityMonitor:
    """
    Unified criticality assessment combining multiple indicators.
    
    Monitors networks for signs of criticality including dead neurons,
    perturbation sensitivity, and decision boundary complexity.
    Used to guide adaptive spectral regularization.
    """
    
    def __init__(self, dead_threshold: float = 1e-5, perturbation_eps: float = 1e-3):
        """
        Initialize criticality monitor.
        
        Args:
            dead_threshold: Activation threshold below which neurons are considered dead
            perturbation_eps: Perturbation magnitude for sensitivity analysis
        """
        self.dead_threshold = dead_threshold
        self.perturbation_eps = perturbation_eps
    
    def assess_criticality(self, 
                          model: SpectralRegularizedModel, 
                          data: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive criticality metrics.
        
        Args:
            model: Model to analyze
            data: Sample data for analysis
            
        Returns:
            Dictionary containing:
            - dead_neuron_rate: Fraction of neurons with low activation
            - perturbation_sensitivity: Response to input perturbations
            - spectral_radius_avg: Average top singular value across layers
            - boundary_fractal_dim: Decision boundary complexity (if applicable)
            
        Note:
            Some metrics may raise NotImplementedError in Phase 1.
            Implementation will be completed incrementally.
        """
        model.eval()
        
        with torch.no_grad():
            # Get preactivations for dead neuron analysis
            _, preactivations = model.forward_with_preactivations(data)
            
            # Compute dead neuron rate
            dead_rate = self._compute_dead_neuron_rate(preactivations)
            
            # Compute perturbation sensitivity
            sensitivity = self._compute_perturbation_sensitivity(model, data)
            
            # Compute average spectral radius
            weights = model.get_regularizable_weights()
            spectral_radii = [self._estimate_spectral_radius(w) for w in weights]
            avg_spectral_radius = float(np.mean(spectral_radii)) if spectral_radii else 0.0
            
            # Boundary fractal dimension
            fractal_dim = self._compute_boundary_fractal_dim(model, data)
        
        return {
            'dead_neuron_rate': dead_rate,
            'perturbation_sensitivity': sensitivity,
            'spectral_radius_avg': avg_spectral_radius,
            'boundary_fractal_dim': fractal_dim
        }
    
    def criticality_score(self, metrics: Dict[str, float]) -> float:
        """
        Combine individual metrics into unified criticality score [0,1].
        
        Args:
            metrics: Dictionary of individual criticality metrics
            
        Returns:
            Unified criticality score where ~0.5 indicates edge of chaos
            
        Note:
            Phase 1 implementation uses simple weighted combination.
            Phase 2 will add learned weights and validation.
        """
        # Simple weighted combination - will be refined in Phase 2
        dead_rate = metrics.get('dead_neuron_rate', 0.0)
        sensitivity = metrics.get('perturbation_sensitivity', 0.0)
        spectral_avg = metrics.get('spectral_radius_avg', 0.0)
        
        # Normalize components (approximate ranges)
        dead_norm = min(dead_rate, 1.0)  # Already in [0,1]
        sens_norm = min(sensitivity / 10.0, 1.0)  # Rough normalization
        spec_norm = min(abs(spectral_avg - 1.0), 1.0)  # Distance from target σ=1.0
        
        # Weighted combination (equal weights for Phase 1)
        score = (dead_norm + sens_norm + spec_norm) / 3.0
        return float(score)
    
    def _compute_dead_neuron_rate(self, preactivations: List[torch.Tensor]) -> float:
        """Compute fraction of dead neurons across all layers."""
        if not preactivations:
            return 0.0
        
        dead_rates = []
        for preact in preactivations[:-1]:  # Exclude output layer
            # Mean absolute activation per neuron
            mean_abs = preact.abs().mean(dim=0)
            # Fraction below threshold
            dead_frac = (mean_abs < self.dead_threshold).float().mean().item()
            dead_rates.append(dead_frac)
        
        return float(np.mean(dead_rates)) if dead_rates else 0.0
    
    def _compute_perturbation_sensitivity(self, 
                                        model: SpectralRegularizedModel,
                                        data: torch.Tensor,
                                        n_directions: int = 10) -> float:
        """Compute average sensitivity to input perturbations."""
        # Use subset of data for efficiency
        sample_size = min(128, len(data))
        sample_indices = torch.randperm(len(data))[:sample_size]
        x_sample = data[sample_indices]
        
        with torch.no_grad():
            # Baseline output
            baseline_output = model(x_sample)
            
            sensitivity_sum = 0.0
            for _ in range(n_directions):
                # Random unit perturbation
                perturbation = torch.randn_like(x_sample)
                perturbation = perturbation / (perturbation.norm(dim=1, keepdim=True) + 1e-12)
                perturbation *= self.perturbation_eps
                
                # Perturbed output
                perturbed_output = model(x_sample + perturbation)
                
                # Sensitivity = output change / perturbation magnitude
                output_change = (perturbed_output - baseline_output).norm(dim=1)
                sensitivity = output_change / self.perturbation_eps
                sensitivity_sum += sensitivity.mean().item()
        
        return sensitivity_sum / n_directions
    
    def _estimate_spectral_radius(self, weight_matrix: torch.Tensor, n_iters: int = 10) -> float:
        """
        Estimate top singular value using power iteration.
        
        Note:
            This is extracted from prototypes/SAMPLE-CODE-v1.md
            Provides efficient spectral radius estimation.
        """
        if weight_matrix.dim() != 2:
            return 0.0
        
        W = weight_matrix.detach()
        
        # Initialize random vector
        b = torch.randn(W.shape[1], device=W.device)
        b = b / (b.norm() + 1e-9)
        
        # Power iteration
        for _ in range(n_iters):
            # Forward: v = W @ b
            v = W @ b
            if v.norm() == 0:
                return 0.0
            
            # Backward: b = W^T @ v, normalize
            b = W.t() @ v
            b = b / (b.norm() + 1e-9)
        
        # Final forward pass to get singular value
        v = W @ b
        return v.norm().item() / (b.norm().item() + 1e-12)
    
    def _compute_boundary_fractal_dim(self, 
                                    model: SpectralRegularizedModel,
                                    data: torch.Tensor,
                                    resolution: int = 200) -> float:
        """
        Compute decision boundary fractal dimension using box-counting method.
        
        Extracts decision boundary from model predictions on a grid and computes
        fractal dimension using box-counting analysis.
        
        Args:
            model: Model to analyze
            data: Sample data (used for device placement)
            resolution: Grid resolution for boundary extraction
            
        Returns:
            Fractal dimension of decision boundary (typically 0.5-2.0)
            Returns 0.0 for degenerate cases (uniform predictions, etc.)
        """
        try:
            # Create grid for boundary analysis using standard coordinate bounds
            device = data.device
            x_range = torch.linspace(-1.5, 2.5, resolution, device=device)
            y_range = torch.linspace(-1.0, 1.5, resolution, device=device)
            
            # Create meshgrid and flatten for model evaluation
            grid_x, grid_y = torch.meshgrid(x_range, y_range, indexing='ij')
            grid_points = torch.stack([grid_x.ravel(), grid_y.ravel()], dim=-1)
            
            with torch.no_grad():
                # Get model predictions on grid
                logits = model(grid_points)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                
                # Convert to probabilities and reshape to grid
                probs = torch.sigmoid(logits).cpu().numpy().reshape(resolution, resolution)
                
                # Extract boundary using gradient analysis
                boundary_mask = self._extract_boundary_mask(probs)
                
                # Compute fractal dimension using box-counting
                fractal_dim = self._box_counting_fractal_dim(boundary_mask)
                
                return float(fractal_dim)
                
        except Exception:
            # Return 0.0 for any errors (GPU memory, shape mismatches, etc.)
            return 0.0
    
    def _extract_boundary_mask(self, prob_grid: np.ndarray) -> np.ndarray:
        """
        Extract decision boundary from probability grid using gradient analysis.
        
        Args:
            prob_grid: 2D array of model probabilities
            
        Returns:
            Binary mask indicating boundary regions
        """
        # Compute gradient magnitude
        grad_x, grad_y = np.gradient(prob_grid)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Threshold at 99th percentile to extract boundary
        threshold = np.percentile(grad_magnitude, 99)
        boundary_mask = (grad_magnitude > threshold).astype(np.uint8)
        
        return boundary_mask
    
    def _box_counting_fractal_dim(self, boundary_img: np.ndarray, 
                                box_sizes: Optional[List[int]] = None) -> float:
        """
        Compute fractal dimension using box-counting method.
        
        Args:
            boundary_img: Binary image of boundary
            box_sizes: List of box sizes for counting (default: [1,2,4,8,16,32,64])
            
        Returns:
            Fractal dimension estimated from box-counting
        """
        if box_sizes is None:
            box_sizes = [1, 2, 4, 8, 16, 32, 64]
        
        H, W = boundary_img.shape
        counts = []
        
        for box_size in box_sizes:
            # Number of boxes in each dimension
            n_boxes_h = int(np.ceil(H / box_size))
            n_boxes_w = int(np.ceil(W / box_size))
            
            count = 0
            for i in range(n_boxes_h):
                for j in range(n_boxes_w):
                    # Extract box region
                    y_slice = slice(i * box_size, min((i + 1) * box_size, H))
                    x_slice = slice(j * box_size, min((j + 1) * box_size, W))
                    box_region = boundary_img[y_slice, x_slice]
                    
                    # Count box if it contains any boundary pixels
                    if box_region.any():
                        count += 1
            
            counts.append(count if count > 0 else 1)
        
        # Compute fractal dimension from log-log slope
        try:
            # Convert to log-log coordinates
            log_box_sizes = np.log(1.0 / (np.array(box_sizes) / max(H, W)))
            log_counts = np.log(np.array(counts))
            
            # Fit line and extract slope (fractal dimension)
            slope, _ = np.polyfit(log_box_sizes, log_counts, 1)
            
            return slope
            
        except (np.linalg.LinAlgError, ValueError):
            # Return 0.0 for degenerate cases
            return 0.0