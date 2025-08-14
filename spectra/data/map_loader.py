"""
Belgium-Netherlands boundary map loader for neural network training.

This module provides data loading for the Baarle-Nassau/Baarle-Hertog border
classification task, creating raster data suitable for boundary learning experiments.

Migrated from prototypes/map_loader.py with interface standardization and
proper error handling following ARCHITECTURE.md patterns.
"""

import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Tuple, Optional, Union, Dict, Any
import warnings

try:
    import cairosvg
    CAIRO_AVAILABLE = True
except ImportError:
    CAIRO_AVAILABLE = False

try:
    from PIL import Image, ImageDraw
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class BaarleMapLoader:
    """
    Loader for the Belgium-Netherlands border map dataset.
    
    Provides high-quality rasterization of the complex Baarle-Nassau/Baarle-Hertog
    border for neural network boundary learning experiments. Supports configurable
    resolution and coordinate bounds matching the SPECTRA experimental setup.
    
    Color Encoding:
        Belgium: #ffe912 (yellow) → label 1
        Netherlands: #ffffde (light cream) → label 0
    
    Standard Coordinate Bounds: (-1.5, 2.5, -1.0, 1.5)
    """
    
    def __init__(self, svg_path: Optional[Union[str, Path]] = None):
        """
        Initialize the map loader.
        
        Args:
            svg_path: Path to SVG file. If None, uses package default location.
            
        Raises:
            FileNotFoundError: If SVG file not found
            ImportError: If required dependencies missing for SVG processing
        """
        if svg_path is None:
            # Default to package data location
            package_dir = Path(__file__).parent
            svg_path = package_dir / "Baarle-Nassau_-_Baarle-Hertog-en.svg"
        
        self.svg_path = Path(svg_path)
        if not self.svg_path.exists():
            raise FileNotFoundError(
                f"SVG file not found: {svg_path}. "
                f"Expected at: {self.svg_path.absolute()}"
            )
        
        # Parse SVG metadata
        self._parse_svg_info()
        
        # Color encoding (validated from SVG analysis)
        self.belgium_color = "#ffe912"  # Yellow
        self.netherlands_color = "#ffffde"  # Light cream
        self.belgium_rgb = np.array([255, 233, 18])
        self.netherlands_rgb = np.array([255, 255, 222])
        
        # Standard coordinate bounds for SPECTRA experiments
        self.default_bounds = (-1.5, 2.5, -1.0, 1.5)
    
    def _parse_svg_info(self) -> None:
        """Parse SVG file to extract dimensions and validate format."""
        try:
            tree = ET.parse(self.svg_path)
            root = tree.getroot()
            
            # Extract dimensions
            self.svg_width = float(root.get('width', 1700))
            self.svg_height = float(root.get('height', 1700))
            
        except Exception as e:
            warnings.warn(f"Could not parse SVG metadata: {e}")
            # Use default dimensions based on known file
            self.svg_width = 1700.0
            self.svg_height = 1700.0
    
    def get_grid_data(self, 
                     resolution: int = 200,
                     bounds: Optional[Tuple[float, float, float, float]] = None,
                     return_coordinates: bool = True) -> Union[Tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
        """
        Generate training data grid at specified resolution and bounds.
        
        Args:
            resolution: Grid resolution (creates resolution x resolution grid)
            bounds: (x_min, x_max, y_min, y_max). If None, uses default bounds.
            return_coordinates: If True, return coordinate grids as well
            
        Returns:
            If return_coordinates=True: (grid_x, grid_y, labels)
            If return_coordinates=False: labels only
            
            grid_x, grid_y: Coordinate meshgrids of shape (resolution, resolution)
            labels: Binary classification labels of shape (resolution, resolution)
        """
        if bounds is None:
            bounds = self.default_bounds
        
        x_min, x_max, y_min, y_max = bounds
        
        # Create coordinate grid
        x_coords = np.linspace(x_min, x_max, resolution)
        y_coords = np.linspace(y_min, y_max, resolution)
        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        
        # Generate raster image and convert to labels
        raster_image = self._rasterize_svg(resolution, resolution)
        labels = self._raster_to_labels(raster_image)
        
        if return_coordinates:
            return grid_x, grid_y, labels
        else:
            return labels
    
    def get_torch_tensors(self, 
                         resolution: int = 200,
                         bounds: Optional[Tuple[float, float, float, float]] = None,
                         device: Optional[Union[str, torch.device]] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data as PyTorch tensors for neural network training.
        
        Args:
            resolution: Grid resolution
            bounds: Coordinate bounds. If None, uses default bounds.
            device: Target device ('cpu', 'cuda', etc.). If None, uses default.
            
        Returns:
            (coordinates, labels) where:
            - coordinates: Tensor of shape (resolution^2, 2) with (x, y) coordinates
            - labels: Tensor of shape (resolution^2,) with binary labels (long type)
            
        Raises:
            ImportError: If PyTorch not available
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for tensor output. Install with: pip install torch")
        
        grid_x, grid_y, labels = self.get_grid_data(resolution, bounds, return_coordinates=True)
        
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
    
    def _rasterize_svg(self, width: int, height: int) -> np.ndarray:
        """
        Convert SVG to raster image at specified resolution.
        
        Args:
            width: Output image width in pixels
            height: Output image height in pixels
            
        Returns:
            RGB image as numpy array of shape (height, width, 3)
        """
        if CAIRO_AVAILABLE and PIL_AVAILABLE:
            return self._rasterize_cairo(width, height)
        else:
            # Graceful degradation with warning
            warnings.warn(
                "cairosvg and/or PIL not available. Install with: "
                "pip install cairosvg pillow. Using fallback rasterization."
            )
            return self._rasterize_fallback(width, height)
    
    def _rasterize_cairo(self, width: int, height: int) -> np.ndarray:
        """High-quality SVG rasterization using cairosvg."""
        try:
            # Convert SVG to PNG bytes
            png_bytes = cairosvg.svg2png(
                url=str(self.svg_path),
                output_width=width,
                output_height=height
            )
            
            # Convert to numpy array
            from io import BytesIO
            image = Image.open(BytesIO(png_bytes))
            
            # Convert RGBA to RGB if necessary
            if image.mode == 'RGBA':
                # Create white background
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                image = rgb_image
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
            
        except Exception as e:
            warnings.warn(f"Cairo rasterization failed: {e}. Using fallback.")
            return self._rasterize_fallback(width, height)
    
    def _rasterize_fallback(self, width: int, height: int) -> np.ndarray:
        """
        Fallback rasterization creating a complex boundary pattern.
        
        Note:
            This is NOT the real map data, but provides a complex boundary
            structure for testing when proper SVG libraries aren't available.
            Used only as development fallback.
        """
        warnings.warn(
            "Using fallback rasterization - not real map data! "
            "Install cairosvg and pillow for authentic boundary data."
        )
        
        # Create complex pattern resembling fragmented border structure
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        y_coords, x_coords = np.mgrid[0:height, 0:width]
        x_norm = x_coords / width
        y_norm = y_coords / height
        
        # Multi-frequency pattern creating fragmented regions
        pattern1 = np.sin(x_norm * 8 * np.pi) * np.cos(y_norm * 6 * np.pi)
        pattern2 = np.sin((x_norm + y_norm) * 10 * np.pi)
        pattern3 = np.sin(x_norm * 15 * np.pi) * np.sin(y_norm * 12 * np.pi)
        
        combined = pattern1 + 0.3 * pattern2 + 0.2 * pattern3
        belgium_mask = combined > 0.1
        
        # Add some enclave-like structures
        np.random.seed(42)  # Deterministic fallback pattern
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
        image[belgium_mask] = self.belgium_rgb
        image[~belgium_mask] = self.netherlands_rgb
        
        return image
    
    def _raster_to_labels(self, raster_image: np.ndarray) -> np.ndarray:
        """
        Convert raster image to binary labels based on color classification.
        
        Args:
            raster_image: RGB image array of shape (height, width, 3)
            
        Returns:
            Binary label array of shape (height, width) where:
            1 = Belgium (yellow regions)
            0 = Netherlands (cream regions)
        """
        # Calculate color distances
        belgium_dist = np.sum((raster_image - self.belgium_rgb[None, None, :]) ** 2, axis=2)
        netherlands_dist = np.sum((raster_image - self.netherlands_rgb[None, None, :]) ** 2, axis=2)
        
        # Assign labels based on closest color
        labels = (belgium_dist < netherlands_dist).astype(np.uint8)
        
        return labels
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary information about the loaded map and capabilities.
        
        Returns:
            Dictionary with loader configuration and dependency status
        """
        return {
            'svg_path': str(self.svg_path),
            'svg_dimensions': (self.svg_width, self.svg_height),
            'belgium_color': self.belgium_color,
            'netherlands_color': self.netherlands_color,
            'default_bounds': self.default_bounds,
            'dependencies': {
                'cairosvg_available': CAIRO_AVAILABLE,
                'pil_available': PIL_AVAILABLE,
                'torch_available': TORCH_AVAILABLE
            }
        }