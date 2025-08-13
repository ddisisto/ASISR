# Baarle Map Loader

A Python loader component for the Belgium-Netherlands border map SVG file, designed for neural network boundary learning experiments based on the Welch Labs visualization.

## Overview

This loader converts the complex Baarle-Nassau/Baarle-Hertog border SVG into raster data suitable for neural network training. It handles coordinate transformations and provides data in formats compatible with PyTorch and the existing experiment code from SAMPLE-CODE-v1.md.

## Quick Start

```python
from map_loader import BaarleMapLoader

# Initialize loader
loader = BaarleMapLoader('Baarle-Nassau_-_Baarle-Hertog-en.svg')

# Get training data (matches SAMPLE-CODE-v1.md format)
grid_x, grid_y, labels = loader.get_grid(200, bounds=(-1.5, 2.5, -1.0, 1.5))

# Get visualization background
background = loader.get_background_image(400)

# Get PyTorch tensors for neural network training
coords_tensor, labels_tensor = loader.get_torch_tensors(200, bounds=(-1.5, 2.5, -1.0, 1.5))
```

## Installation

### Required Dependencies
```bash
pip install numpy
```

### Recommended Dependencies (for full functionality)
```bash
pip install cairosvg Pillow matplotlib torch
# Or install all at once:
pip install -r requirements_map_loader.txt
```

## Files

- `map_loader.py` - Main loader component
- `example_usage.py` - Complete usage examples and integration guide  
- `requirements_map_loader.txt` - Dependencies list
- `Baarle-Nassau_-_Baarle-Hertog-en.svg` - Source map file

## Color Encoding

Based on analysis of the SVG file:
- **Belgium regions**: `#ffe912` (yellow) → Label = 1
- **Netherlands regions**: `#ffffde` (light cream) → Label = 0

## Coordinate System

The loader supports the coordinate system from SAMPLE-CODE-v1.md:
- Default bounds: `(-1.5, 2.5, -1.0, 1.5)` (x_min, x_max, y_min, y_max)
- Coordinate ranges suitable for neural network input normalization
- Grid generation: `np.meshgrid(np.linspace(-1.5,2.5,200), np.linspace(-1.0,1.5,200))`

## Key Features

1. **SVG Parsing**: Handles complex SVG with multiple path elements
2. **Rasterization**: High-quality conversion using CairoSVG (with fallback)
3. **Coordinate Transformation**: Maps geographic data to neural network coordinate space
4. **Multiple Output Formats**: NumPy arrays, PyTorch tensors, visualization images
5. **Resolution Flexibility**: Generate data at any resolution on demand
6. **Boundary Detection**: Compatible with gradient-based boundary analysis

## Integration with Existing Code

Replace the two-moons dataset in SAMPLE-CODE-v1.md:

```python
# Original code:
# (X_train, y_train), (X_test, y_test) = make_dataset(seed=seed)

# New code:
from map_loader import BaarleMapLoader

def make_baarle_dataset(resolution=200, test_size=0.25, seed=0):
    loader = BaarleMapLoader('Baarle-Nassau_-_Baarle-Hertog-en.svg')
    coords, labels = loader.get_torch_tensors(resolution, (-1.5, 2.5, -1.0, 1.5))
    
    # Add some noise for robustness (optional)
    torch.manual_seed(seed)
    coords += torch.randn_like(coords) * 0.01
    
    # Split into train/test
    n_samples = len(coords)
    indices = torch.randperm(n_samples)
    split_idx = int(n_samples * (1 - test_size))
    
    return ((coords[indices[:split_idx]], labels[indices[:split_idx]]), 
            (coords[indices[split_idx:]], labels[indices[split_idx:]]))

# Use in experiment:
(X_train, y_train), (X_test, y_test) = make_baarle_dataset(seed=seed)
```

## Visualization and Analysis

The loader provides visualization tools compatible with the boundary analysis from SAMPLE-CODE-v1.md:

```python
# Visualize the map and labels
loader.visualize_grid(resolution=200, bounds=(-1.5, 2.5, -1.0, 1.5))

# Boundary detection (for fractal dimension analysis)
grid_x, grid_y, labels = loader.get_grid(200, bounds=(-1.5, 2.5, -1.0, 1.5))
gy, gx = np.gradient(labels.astype(float))
grad_magnitude = np.sqrt(gx**2 + gy**2)
boundary_mask = grad_magnitude > np.percentile(grad_magnitude, 99)
```

## Error Handling and Fallbacks

- **Missing CairoSVG**: Falls back to simple pattern generation for testing
- **Missing PIL**: Uses alternative image handling
- **Missing matplotlib**: Disables visualization features gracefully
- **Missing PyTorch**: Disables tensor functionality with clear error messages

## Example Output

- **Grid data**: Matches the `np.meshgrid` format from SAMPLE-CODE-v1.md
- **Labels**: Binary classification (0/1) suitable for `BCEWithLogitsLoss`
- **Coordinates**: Normalized to specified bounds for neural network training
- **Resolution**: Any resolution supported (tested with 50x50 to 1000x1000)

## Notes

- The actual SVG is extremely detailed with complex polygonal boundaries
- Fallback pattern provides complex boundary structure for testing when SVG libraries aren't available
- Color detection is robust to minor variations in SVG encoding
- Compatible with the spectral regularization experiment from SAMPLE-CODE-v1.md

## Source Map

The loader is designed for use with the specific SVG file:
- **File**: `Baarle-Nassau_-_Baarle-Hertog-en.svg`
- **Source**: Wikimedia Commons
- **License**: Public domain
- **Dimensions**: 1700x1700 pixels

This represents one of the most complex international borders in the world, making it ideal for testing neural network boundary learning capabilities.