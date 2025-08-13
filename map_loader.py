"""
Baarle-Nassau/Baarle-Hertog Map Loader for Neural Network Training

This module provides a loader for the Belgium-Netherlands border map SVG file
that creates raster data suitable for neural network boundary learning experiments.

Based on the Welch Labs visualization from DL-TRANSCRIPT.md, this loader can generate
training data at any resolution and handle coordinate transformations for neural networks.

Author: Generated for ASISR boundary learning experiment
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings

try:
    import cairosvg
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False
    warnings.warn("cairosvg not available. Install with: pip install cairosvg")

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    warnings.warn("PIL not available. Install with: pip install Pillow")

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("matplotlib not available. Install with: pip install matplotlib")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaarleMapLoader:
    """
    Loader for the Baarle-Nassau/Baarle-Hertog border map.
    
    This class can parse the SVG file and generate raster data at any resolution,
    suitable for neural network training on complex decision boundaries.
    
    Color encoding from SVG analysis:
    - #ffe912 (yellow): One country (assigned value 1)  
    - #ffffde (light cream): Other country (assigned value 0)
    
    Attributes:
        svg_path (Path): Path to the SVG file
        svg_width (float): Original SVG width in pixels
        svg_height (float): Original SVG height in pixels
        belgium_color (str): Hex color for Belgium regions
        netherlands_color (str): Hex color for Netherlands regions
    """
    
    def __init__(self, svg_path: Union[str, Path]):
        """
        Initialize the map loader.
        
        Args:
            svg_path: Path to the Baarle SVG file
        """
        self.svg_path = Path(svg_path)
        if not self.svg_path.exists():
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
        
        # Parse SVG to get dimensions
        self._parse_svg_info()
        
        # Color mapping based on analysis of the SVG file
        # #ffe912 appears to be Belgium (yellow)
        # #ffffde appears to be Netherlands (light cream)
        self.belgium_color = "#ffe912"
        self.netherlands_color = "#ffffde"
        
        print(f"Loaded SVG: {self.svg_path.name}")
        print(f"Dimensions: {self.svg_width} x {self.svg_height}")
        print(f"Belgium color: {self.belgium_color}")
        print(f"Netherlands color: {self.netherlands_color}")
    
    def _parse_svg_info(self):
        """Parse SVG file to extract dimensions and basic info."""
        try:
            tree = ET.parse(self.svg_path)
            root = tree.getroot()
            
            # Extract width and height from SVG root element
            self.svg_width = float(root.get('width', 1700))
            self.svg_height = float(root.get('height', 1700))
            
        except Exception as e:
            warnings.warn(f"Could not parse SVG dimensions: {e}")
            # Default dimensions based on the file we examined
            self.svg_width = 1700.0
            self.svg_height = 1700.0
    
    def svg_to_raster(self, width: int, height: int, method: str = 'cairo') -> np.ndarray:
        """
        Convert SVG to raster image at specified resolution.
        
        Args:
            width: Output image width in pixels
            height: Output image height in pixels  
            method: Rasterization method ('cairo' or 'simple')
            
        Returns:
            RGB image as numpy array of shape (height, width, 3)
        """
        if method == 'cairo' and CAIRO_AVAILABLE:
            return self._svg_to_raster_cairo(width, height)
        else:
            return self._svg_to_raster_simple(width, height)
    
    def _svg_to_raster_cairo(self, width: int, height: int) -> np.ndarray:
        """Use cairosvg for high-quality SVG rasterization."""
        try:
            # Convert SVG to PNG bytes
            png_bytes = cairosvg.svg2png(
                url=str(self.svg_path),
                output_width=width,
                output_height=height
            )
            
            # Convert to PIL Image then numpy
            if PIL_AVAILABLE:
                from io import BytesIO
                image = Image.open(BytesIO(png_bytes))
                return np.array(image)
            else:
                raise ImportError("PIL required for cairo method")
                
        except Exception as e:
            warnings.warn(f"Cairo rasterization failed: {e}. Falling back to simple method.")
            return self._svg_to_raster_simple(width, height)
    
    def _svg_to_raster_simple(self, width: int, height: int) -> np.ndarray:
        """
        Simple rasterization method that creates a placeholder pattern.
        This is a fallback when proper SVG parsing libraries aren't available.
        """
        warnings.warn("Using simple rasterization - install cairosvg for proper SVG rendering")
        
        # Create a simple pattern that resembles the complex boundary structure
        # This is obviously not the real map, but provides a complex boundary for testing
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        # Create some complex boundary patterns similar to the actual map
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        
        # Normalize coordinates to [0, 1]
        x_norm = x_coords / width
        y_norm = y_coords / height
        
        # Create complex pattern with multiple regions
        # This approximates the fragmented nature of the Baarle border
        pattern1 = np.sin(x_norm * 8 * np.pi) * np.cos(y_norm * 6 * np.pi)
        pattern2 = np.sin((x_norm + y_norm) * 10 * np.pi) 
        pattern3 = np.sin(x_norm * 15 * np.pi) * np.sin(y_norm * 12 * np.pi)
        
        # Combine patterns to create fragmented regions
        combined = (pattern1 + 0.3 * pattern2 + 0.2 * pattern3)
        
        # Create binary mask with some islands and enclaves
        belgium_mask = combined > 0.1
        
        # Add some smaller enclaves
        for _ in range(20):
            cx = np.random.randint(width//4, 3*width//4)
            cy = np.random.randint(height//4, 3*height//4)
            r = np.random.randint(5, 30)
            
            circle_mask = ((x_coords - cx)**2 + (y_coords - cy)**2) < r**2
            if np.random.random() > 0.5:
                belgium_mask[circle_mask] = True
            else:
                belgium_mask[circle_mask] = False
        
        # Apply colors
        # Belgium regions: yellow (#ffe912)
        image[belgium_mask] = [255, 233, 18]  # Belgium yellow
        # Netherlands regions: light cream (#ffffde) 
        image[~belgium_mask] = [255, 255, 222]  # Netherlands cream
        
        return image
    
    def raster_to_labels(self, raster_image: np.ndarray) -> np.ndarray:
        """
        Convert raster image to binary labels based on color.
        
        Args:
            raster_image: RGB image array of shape (height, width, 3)
            
        Returns:
            Binary label array of shape (height, width) where:
            - 1 = Belgium (yellow regions)
            - 0 = Netherlands (cream regions)
        """
        # Convert hex colors to RGB
        belgium_rgb = np.array([255, 233, 18])  # #ffe912
        netherlands_rgb = np.array([255, 255, 222])  # #ffffde
        
        # Calculate distances to each color
        belgium_dist = np.sum((raster_image - belgium_rgb[None, None, :]) ** 2, axis=2)
        netherlands_dist = np.sum((raster_image - netherlands_rgb[None, None, :]) ** 2, axis=2)
        
        # Assign labels based on closest color
        labels = (belgium_dist < netherlands_dist).astype(np.uint8)
        
        return labels
    
    def get_grid(self, resolution: int = 200, 
                 bounds: Tuple[float, float, float, float] = (-1.5, 2.5, -1.0, 1.5),
                 return_coordinates: bool = True) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Generate a grid of training data at specified resolution and bounds.
        
        This matches the interface from SAMPLE-CODE-v1.md:
        grid_x, grid_y = np.meshgrid(np.linspace(-1.5,2.5,200), np.linspace(-1.0,1.5,200))
        
        Args:
            resolution: Grid resolution (creates resolution x resolution grid)
            bounds: (x_min, x_max, y_min, y_max) coordinate bounds
            return_coordinates: If True, return coordinate grids as well
            
        Returns:
            If return_coordinates=True: (grid_x, grid_y, labels)
            If return_coordinates=False: labels only
            
            - grid_x, grid_y: Coordinate meshgrids of shape (resolution, resolution)
            - labels: Binary labels of shape (resolution, resolution)
        """
        x_min, x_max, y_min, y_max = bounds
        
        # Create coordinate grid matching the sample code format
        x_coords = np.linspace(x_min, x_max, resolution)
        y_coords = np.linspace(y_min, y_max, resolution)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        
        # Generate raster image at this resolution
        raster = self.svg_to_raster(resolution, resolution)
        
        # Convert to binary labels
        labels = self.raster_to_labels(raster)
        
        if return_coordinates:
            return grid_x, grid_y, labels
        else:
            return labels
    
    def get_background_image(self, resolution: int = 400) -> np.ndarray:
        """
        Get a high-resolution background image for visualization.
        
        Args:
            resolution: Image resolution for background
            
        Returns:
            RGB image array of shape (resolution, resolution, 3)
        """
        return self.svg_to_raster(resolution, resolution)
    
    def visualize_grid(self, resolution: int = 200, 
                      bounds: Tuple[float, float, float, float] = (-1.5, 2.5, -1.0, 1.5),
                      figsize: Tuple[int, int] = (10, 8)):
        """
        Visualize the grid data with matplotlib.
        
        Args:
            resolution: Grid resolution
            bounds: Coordinate bounds  
            figsize: Figure size for matplotlib
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("matplotlib required for visualization")
        
        grid_x, grid_y, labels = self.get_grid(resolution, bounds)
        background = self.get_background_image(resolution)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Show original map
        ax1.imshow(background, extent=bounds, origin='lower', alpha=0.8)
        ax1.set_title("Original Map")
        ax1.set_xlabel("X coordinate")
        ax1.set_ylabel("Y coordinate")
        ax1.grid(True, alpha=0.3)
        
        # Show binary labels
        im2 = ax2.imshow(labels, extent=bounds, origin='lower', cmap='RdYlBu_r')
        ax2.set_title("Binary Labels\n(Yellow=Belgium=1, Blue=Netherlands=0)")
        ax2.set_xlabel("X coordinate") 
        ax2.set_ylabel("Y coordinate")
        ax2.grid(True, alpha=0.3)
        
        plt.colorbar(im2, ax=ax2, label="Country Label")
        plt.tight_layout()
        plt.show()
    
    def get_torch_tensors(self, resolution: int = 200,
                         bounds: Tuple[float, float, float, float] = (-1.5, 2.5, -1.0, 1.5),
                         device: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data as PyTorch tensors for neural network training.
        
        Args:
            resolution: Grid resolution
            bounds: Coordinate bounds
            device: Target device ('cpu', 'cuda', etc.)
            
        Returns:
            (coordinates, labels) where:
            - coordinates: Tensor of shape (resolution^2, 2) with (x, y) coordinates
            - labels: Tensor of shape (resolution^2,) with binary labels
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        
        grid_x, grid_y, labels = self.get_grid(resolution, bounds)
        
        # Flatten and stack coordinates
        coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=-1)
        labels_flat = labels.ravel()
        
        # Convert to tensors
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_flat, dtype=torch.long)
        
        # Move to device if specified
        if device is not None:
            coords_tensor = coords_tensor.to(device)
            labels_tensor = labels_tensor.to(device)
        
        return coords_tensor, labels_tensor
    
    def summary(self) -> dict:
        """
        Get summary information about the loaded map.
        
        Returns:
            Dictionary with map information
        """
        return {
            'svg_path': str(self.svg_path),
            'svg_dimensions': (self.svg_width, self.svg_height),
            'belgium_color': self.belgium_color,
            'netherlands_color': self.netherlands_color,
            'cairo_available': CAIRO_AVAILABLE,
            'pil_available': PIL_AVAILABLE,
            'matplotlib_available': MATPLOTLIB_AVAILABLE,
            'torch_available': TORCH_AVAILABLE
        }


def demo_usage():
    """
    Demonstration of the BaarleMapLoader usage.
    """
    print("=== Baarle Map Loader Demo ===\n")
    
    # Initialize loader
    try:
        loader = BaarleMapLoader('Baarle-Nassau_-_Baarle-Hertog-en.svg')
    except FileNotFoundError:
        print("SVG file not found. Please ensure 'Baarle-Nassau_-_Baarle-Hertog-en.svg' is in the current directory.")
        return
    
    # Print summary
    print("\nMap Summary:")
    summary = loader.summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Generate training data (matching SAMPLE-CODE-v1.md format)
    print("\n=== Generating Training Data ===")
    resolution = 200
    bounds = (-1.5, 2.5, -1.0, 1.5)
    
    grid_x, grid_y, labels = loader.get_grid(resolution, bounds)
    print(f"Grid shape: {grid_x.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Belgium regions (label=1): {(labels == 1).sum()} pixels")
    print(f"Netherlands regions (label=0): {(labels == 0).sum()} pixels")
    
    # Get background image
    background = loader.get_background_image(400)
    print(f"Background image shape: {background.shape}")
    
    # Show visualization if matplotlib available
    if MATPLOTLIB_AVAILABLE:
        print("\n=== Visualization ===")
        try:
            loader.visualize_grid(resolution=100, bounds=bounds, figsize=(12, 5))
        except Exception as e:
            print(f"Visualization failed: {e}")
    
    # PyTorch tensors if available
    if TORCH_AVAILABLE:
        print("\n=== PyTorch Tensors ===")
        try:
            coords, labels_torch = loader.get_torch_tensors(resolution=50, bounds=bounds)
            print(f"Coordinates tensor shape: {coords.shape}")
            print(f"Labels tensor shape: {labels_torch.shape}")
            print(f"Coordinates range: x=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
                  f"y=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}]")
        except Exception as e:
            print(f"PyTorch tensors failed: {e}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    demo_usage()