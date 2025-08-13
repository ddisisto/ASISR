#!/usr/bin/env python3
"""
Example usage of the Baarle Map Loader for neural network training.

This demonstrates how to use the map loader for the boundary learning experiment
described in the ASISR project.
"""

import numpy as np
import matplotlib.pyplot as plt
from map_loader import BaarleMapLoader

def main():
    """Main example demonstrating map loader usage."""
    
    print("Loading Baarle-Nassau/Baarle-Hertog border map...")
    
    # Initialize the map loader
    loader = BaarleMapLoader('Baarle-Nassau_-_Baarle-Hertog-en.svg')
    
    # ================================================================
    # Example 1: Generate training data (matches SAMPLE-CODE-v1.md)
    # ================================================================
    print("\n1. Generating training data at 200x200 resolution...")
    
    # This exactly matches the grid generation in SAMPLE-CODE-v1.md:
    # grid_x, grid_y = np.meshgrid(np.linspace(-1.5,2.5,200), np.linspace(-1.0,1.5,200))
    grid_x, grid_y, labels = loader.get_grid(
        resolution=200, 
        bounds=(-1.5, 2.5, -1.0, 1.5)
    )
    
    print(f"  Grid coordinates shape: {grid_x.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Belgium coverage: {(labels == 1).mean()*100:.1f}%")
    print(f"  Netherlands coverage: {(labels == 0).mean()*100:.1f}%")
    
    # ================================================================
    # Example 2: Higher resolution background for visualization
    # ================================================================
    print("\n2. Creating high-resolution background...")
    
    background = loader.get_background_image(resolution=400)
    print(f"  Background image shape: {background.shape}")
    
    # ================================================================
    # Example 3: PyTorch tensors for neural network training
    # ================================================================
    print("\n3. Generating PyTorch tensors...")
    
    try:
        import torch
        
        # Generate tensor data for neural network training
        coords_tensor, labels_tensor = loader.get_torch_tensors(
            resolution=200,
            bounds=(-1.5, 2.5, -1.0, 1.5)
        )
        
        print(f"  Coordinates tensor: {coords_tensor.shape} {coords_tensor.dtype}")
        print(f"  Labels tensor: {labels_tensor.shape} {labels_tensor.dtype}")
        print(f"  Device: {coords_tensor.device}")
        
        # Example: Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        coords_gpu, labels_gpu = loader.get_torch_tensors(
            resolution=100, 
            bounds=(-1.5, 2.5, -1.0, 1.5),
            device=device
        )
        print(f"  GPU tensors device: {coords_gpu.device}")
        
    except ImportError:
        print("  PyTorch not available - skipping tensor example")
    
    # ================================================================
    # Example 4: Visualization and analysis
    # ================================================================
    print("\n4. Creating visualization...")
    
    try:
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original background image
        axes[0, 0].imshow(background, extent=(-1.5, 2.5, -1.0, 1.5), origin='lower')
        axes[0, 0].set_title("Original Map (400x400)")
        axes[0, 0].set_xlabel("X coordinate")
        axes[0, 0].set_ylabel("Y coordinate")
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Binary labels
        im1 = axes[0, 1].imshow(labels, extent=(-1.5, 2.5, -1.0, 1.5), 
                               origin='lower', cmap='RdYlBu_r')
        axes[0, 1].set_title("Binary Labels (200x200)\nYellow=Belgium=1, Blue=Netherlands=0")
        axes[0, 1].set_xlabel("X coordinate")
        axes[0, 1].set_ylabel("Y coordinate")
        axes[0, 1].grid(True, alpha=0.3)
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Plot 3: Boundary detection (gradient magnitude)
        # This is similar to the boundary detection in SAMPLE-CODE-v1.md
        gy, gx = np.gradient(labels.astype(float))
        grad_magnitude = np.sqrt(gx**2 + gy**2)
        
        im2 = axes[1, 0].imshow(grad_magnitude, extent=(-1.5, 2.5, -1.0, 1.5),
                               origin='lower', cmap='hot')
        axes[1, 0].set_title("Boundary Gradient Magnitude")
        axes[1, 0].set_xlabel("X coordinate")
        axes[1, 0].set_ylabel("Y coordinate")
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Plot 4: Boundary mask (for fractal dimension analysis)
        # This matches the boundary detection from SAMPLE-CODE-v1.md
        boundary_mask = grad_magnitude > np.percentile(grad_magnitude, 95)
        
        axes[1, 1].imshow(boundary_mask, extent=(-1.5, 2.5, -1.0, 1.5),
                         origin='lower', cmap='gray')
        axes[1, 1].set_title("Boundary Mask (95th percentile)")
        axes[1, 1].set_xlabel("X coordinate")
        axes[1, 1].set_ylabel("Y coordinate")
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('baarle_map_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("  Visualization saved as 'baarle_map_analysis.png'")
        
    except ImportError:
        print("  Matplotlib not available - skipping visualization")
    
    # ================================================================
    # Example 5: Integration with the existing experiment code
    # ================================================================
    print("\n5. Integration example for neural network training...")
    
    # This shows how to integrate with the code from SAMPLE-CODE-v1.md
    print("  Example code snippet for integration:")
    print("""
    # Replace the make_moons dataset with Baarle map data:
    
    from map_loader import BaarleMapLoader
    
    def make_baarle_dataset(resolution=200, bounds=(-1.5, 2.5, -1.0, 1.5), test_size=0.25):
        loader = BaarleMapLoader('Baarle-Nassau_-_Baarle-Hertog-en.svg')
        coords, labels = loader.get_torch_tensors(resolution, bounds)
        
        # Split into train/test
        n_samples = len(coords)
        indices = torch.randperm(n_samples)
        split_idx = int(n_samples * (1 - test_size))
        
        train_coords = coords[indices[:split_idx]]
        train_labels = labels[indices[:split_idx]]
        test_coords = coords[indices[split_idx:]]
        test_labels = labels[indices[split_idx:]]
        
        return (train_coords, train_labels), (test_coords, test_labels)
    
    # Then use in the experiment:
    (X_train, y_train), (X_test, y_test) = make_baarle_dataset()
    """)
    
    print("\nMap loader setup complete! Ready for neural network experiments.")
    
    # Print summary
    summary = loader.summary()
    print(f"\nSummary:")
    print(f"  SVG path: {summary['svg_path']}")
    print(f"  Dimensions: {summary['svg_dimensions']}")
    print(f"  Available libraries: Cairo={summary['cairo_available']}, "
          f"PIL={summary['pil_available']}, PyTorch={summary['torch_available']}")


if __name__ == "__main__":
    main()