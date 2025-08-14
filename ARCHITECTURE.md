# SPECTRA Code Architecture

**Technical Authority**: This document defines all structural decisions, interfaces, and design patterns for the SPECTRA project.

## Package Structure

```
SPECTRA/
├── spectra/                         # Core library package
│   ├── __init__.py                  # Package exports and version
│   ├── models/                      # Neural network architectures
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract interfaces for spectral regularization
│   │   ├── mlp.py                   # Enhanced MLP with spectral hooks
│   │   └── transformer.py           # Future: Attention matrix regularization
│   ├── regularization/              # Spectral regularization methods
│   │   ├── __init__.py
│   │   ├── base.py                  # Abstract regularizer interface
│   │   ├── fixed.py                 # Fixed sigma targeting (σ = 1.0)
│   │   ├── adaptive.py              # Adaptive targeting with criticality feedback
│   │   └── multi_scale.py           # Hierarchical layer-wise regularization
│   ├── metrics/                     # Criticality assessment and analysis
│   │   ├── __init__.py
│   │   ├── criticality.py           # Dead neurons, sensitivity, fractal dimension
│   │   ├── spectral.py              # Singular value analysis and power laws
│   │   └── boundary.py              # Decision boundary analysis tools
│   ├── data/                        # Dataset loading and preprocessing
│   │   ├── __init__.py
│   │   ├── map_loader.py            # Belgium-Netherlands boundary data
│   │   ├── synthetic.py             # Two-moons and other test datasets
│   │   └── Baarle-Nassau_-_Baarle-Hertog-en.svg  # Boundary map data
│   ├── training/                    # Experiment orchestration
│   │   ├── __init__.py
│   │   ├── experiment.py            # Multi-seed experiment runner
│   │   ├── hooks.py                 # Real-time monitoring during training
│   │   └── optimization.py          # Scale-invariant schedulers
│   ├── visualization/               # Analysis and plotting tools
│   │   ├── __init__.py
│   │   ├── boundaries.py            # Decision boundary visualization
│   │   ├── dynamics.py              # Training trajectory analysis
│   │   └── criticality.py           # Criticality indicator plots
│   └── utils/                       # Common utilities
│       ├── __init__.py
│       ├── seed.py                  # Reproducibility management
│       ├── device.py                # GPU/CPU handling
│       └── config.py                # Configuration loading
├── experiments/                     # Phase-specific experiment implementations
│   ├── phase1_boundary_mapping/     # Belgium-Netherlands proof-of-concept
│   ├── phase2_adaptive_spectra/     # Full adaptive system
│   ├── phase3_multi_scale/          # Hierarchical architectures
│   └── notebooks/                   # Analysis and exploration
├── tests/                           # Unit and integration tests
│   ├── test_models/
│   ├── test_regularization/
│   ├── test_metrics/
│   └── test_integration/
├── configs/                         # YAML experiment configurations
│   ├── phase1_baseline.yaml
│   ├── phase1_spectral.yaml
│   └── adaptive_spectra.yaml
├── scripts/                         # Utility and automation scripts
│   ├── run_experiments.py
│   ├── generate_figures.py
│   └── validate_setup.py
├── prototypes/                      # Legacy and experimental code
├── docs/                           # Reference materials and papers
└── requirements.txt                # Package dependencies
```

### **Rationale for Structure**

**Plugin Architecture**: Core components implement abstract interfaces, enabling easy extension of regularization methods, models, and metrics without modifying existing code.

**Phase Isolation**: Experiment directories mirror research phases, allowing independent development and validation of each stage.

**Data Co-location**: Dataset files stored within the package ensure they move with the code and are easily accessible for import.

**Testing Symmetry**: Test structure mirrors package structure for clear correspondence and comprehensive coverage.

## Core Interfaces

### **1. Model Interface**

```python
# spectra/models/base.py
from abc import ABC, abstractmethod
from typing import List, Dict
import torch

class SpectralRegularizedModel(torch.nn.Module, ABC):
    """Base class for models supporting spectral regularization"""
    
    @abstractmethod
    def get_regularizable_weights(self) -> List[torch.Tensor]:
        """Return weight matrices subject to spectral regularization"""
        pass
    
    @abstractmethod
    def forward_with_preactivations(self, x: torch.Tensor) -> tuple:
        """Forward pass returning (output, preactivations) for criticality analysis"""
        pass
    
    def spectral_loss(self, regularizer) -> torch.Tensor:
        """Compute spectral regularization loss using provided regularizer"""
        return regularizer.compute_loss(self.get_regularizable_weights())
```

### **2. Regularization Interface**

```python
# spectra/regularization/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import torch

class SpectralRegularizer(ABC):
    """Base class for spectral regularization methods"""
    
    @abstractmethod
    def compute_loss(self, weight_matrices: List[torch.Tensor]) -> torch.Tensor:
        """Compute regularization loss for given weight matrices"""
        pass
    
    @abstractmethod
    def get_targets(self) -> Dict[int, float]:
        """Return current target sigma values for each layer"""
        pass
    
    def update_targets(self, criticality_metrics: Dict[str, float]) -> None:
        """Update regularization targets based on criticality feedback (optional)"""
        pass
```

### **3. Metrics Interface**

```python
# spectra/metrics/criticality.py
from typing import Dict, List
import torch

class CriticalityMonitor:
    """Unified criticality assessment combining multiple indicators"""
    
    def __init__(self, dead_threshold: float = 1e-5, perturbation_eps: float = 1e-3):
        self.dead_threshold = dead_threshold
        self.perturbation_eps = perturbation_eps
    
    def assess_criticality(self, 
                          model: SpectralRegularizedModel, 
                          data: torch.Tensor) -> Dict[str, float]:
        """
        Compute comprehensive criticality metrics:
        - dead_neuron_rate: Fraction of neurons with low activation
        - perturbation_sensitivity: Response to input perturbations  
        - boundary_fractal_dim: Decision boundary complexity
        - spectral_radius_avg: Average top singular value across layers
        """
        pass
    
    def criticality_score(self, metrics: Dict[str, float]) -> float:
        """Combine individual metrics into unified criticality score [0,1]"""
        pass
```

## Integration Points

### **Model ↔ Regularization Integration**

Models expose `get_regularizable_weights()` returning weight tensors. Regularizers operate on these tensors without needing model-specific knowledge.

```python
# Training loop integration
model = SpectralMLP(...)
regularizer = AdaptiveSpectralRegularizer(...)

for batch in dataloader:
    # Standard forward pass
    output = model(batch)
    task_loss = criterion(output, targets)
    
    # Spectral regularization
    spectral_loss = model.spectral_loss(regularizer)
    total_loss = task_loss + spectral_loss
    
    # Criticality monitoring
    if epoch % monitor_interval == 0:
        metrics = criticality_monitor.assess_criticality(model, validation_data)
        regularizer.update_targets(metrics)
```

### **Experiment ↔ Components Integration**

Experiments compose models, regularizers, and metrics without tight coupling:

```python
# Experiment configuration
config = {
    'model': {'type': 'SpectralMLP', 'hidden_dims': [64, 64]},
    'regularizer': {'type': 'AdaptiveSpectralRegularizer', 'adaptation_rate': 0.01},
    'metrics': {'criticality_monitor': {'dead_threshold': 1e-5}}
}

# Dynamic instantiation
experiment = SPECTRAExperiment(config)
results = experiment.run(n_seeds=5)
```

### **Visualization ↔ Results Integration**

Visualization functions accept standardized result objects:

```python
# Results structure
ExperimentResults = {
    'metrics_history': Dict[str, List[float]],  # Training trajectories
    'final_metrics': Dict[str, float],          # End-state assessment
    'model_states': List[torch.nn.Module],      # Saved models per seed
    'config': Dict                              # Experiment configuration
}

# Visualization usage
visualizer.plot_training_dynamics(results)
visualizer.plot_boundary_comparison(results_baseline, results_spectral)
```

## Technical Design Decisions

### **1. Plugin vs. Inheritance Architecture**
**Decision**: Plugin-based with abstract interfaces  
**Rationale**: Enables independent development of new regularization methods, easy A/B testing, and clear separation of concerns  
**Trade-off**: Slight complexity increase vs. maximum extensibility for research

### **2. Configuration-Driven vs. Code-Driven Experiments**
**Decision**: YAML configuration files with code orchestration  
**Rationale**: Reproducible experiments, easy parameter sweeps, version control of experimental settings  
**Trade-off**: Additional configuration layer vs. scientific rigor and reproducibility

### **3. Real-time vs. Post-hoc Criticality Assessment**
**Decision**: Real-time monitoring with optional adaptation  
**Rationale**: Enables adaptive regularization and early stopping based on criticality indicators  
**Trade-off**: Computational overhead vs. superior training dynamics

### **4. Centralized vs. Distributed Data Storage**
**Decision**: Data files within package structure  
**Rationale**: Ensures data and code travel together, simplifies import paths, reduces setup complexity  
**Trade-off**: Package size increase vs. deployment simplicity

### **5. Monolithic vs. Modular Visualization**
**Decision**: Separate visualization modules per analysis type  
**Rationale**: Research requires diverse plotting capabilities without bloating core training code  
**Trade-off**: Module proliferation vs. specialized, high-quality visualizations

## Performance Considerations

### **Spectral Analysis Overhead**
- **Target**: <10% training time overhead for spectral regularization
- **Strategy**: Efficient singular value estimation via power iteration
- **Optimization**: Cache computations, vectorized operations, optional GPU acceleration

### **Memory Management**
- **Preactivation Storage**: Circular buffers for criticality analysis
- **Model Checkpointing**: Configurable frequency based on experiment duration
- **Gradient Accumulation**: Support for large effective batch sizes on limited memory

### **Scalability Targets**
- **Network Size**: Support up to transformer-scale models (hundreds of millions of parameters)
- **Experiment Scale**: Efficient multi-seed runs (5-10 seeds standard)
- **Data Scale**: Handle large boundary maps and high-resolution grids

## Extension Points

### **New Regularization Methods**
Implement `SpectralRegularizer` interface:
```python
class MyRegularizer(SpectralRegularizer):
    def compute_loss(self, weight_matrices): ...
    def get_targets(self): ...
```

### **New Model Architectures**  
Implement `SpectralRegularizedModel` interface:
```python
class MyModel(SpectralRegularizedModel):
    def get_regularizable_weights(self): ...
    def forward_with_preactivations(self, x): ...
```

### **New Criticality Metrics**
Extend `CriticalityMonitor` with additional assessment methods:
```python
criticality_monitor.add_metric('my_metric', my_metric_function)
```

### **New Visualization Types**
Add modules to `spectra/visualization/` following standard interface:
```python
def plot_my_analysis(results: ExperimentResults, **kwargs) -> matplotlib.Figure:
    ...
```

## Dependencies

### **Core Requirements**
```python
torch>=1.10.0          # Neural network training
numpy>=1.20.0          # Numerical computation
scipy>=1.7.0           # Scientific computing (SVD, optimization)
scikit-learn>=1.0.0    # ML utilities and datasets
matplotlib>=3.5.0      # Basic plotting
pyyaml>=6.0            # Configuration file parsing
```

### **Visualization Extensions**
```python
seaborn>=0.11.0        # Statistical plotting
plotly>=5.0.0          # Interactive visualizations
cairosvg>=2.5.0        # SVG map rendering
pillow>=8.0.0          # Image processing
```

### **Development Tools**
```python
pytest>=6.0.0         # Testing framework
pytest-cov>=2.12.0    # Coverage analysis
black>=21.0.0          # Code formatting
mypy>=0.910            # Type checking
```

### **Optional Research Extensions**
```python
tensorboard>=2.7.0     # Training visualization
wandb>=0.12.0          # Experiment tracking
numba>=0.54.0          # Performance optimization
```

## Integration Strategy

This architecture is designed to absorb the existing prototype code through systematic migration:

1. **Base Classes**: Implement abstract interfaces first
2. **Core Components**: Migrate `prototypes/SAMPLE-CODE-v1.md` logic to modular structure  
3. **Data Integration**: Move `prototypes/map_loader.py` to `spectra/data/`
4. **Experiment Migration**: Convert existing experiment to new orchestration framework
5. **Extension Development**: Add adaptive and multi-scale features using plugin architecture

---

**Cross-References**:
- Implementation roadmap and research phases: → [PROJECT_PLAN.md](./PROJECT_PLAN.md)
- Development workflow and Claude Code usage: → [CLAUDE.md](./CLAUDE.md)
- Legacy code and prototypes: → [prototypes/](./prototypes/)
- Research context and papers: → [docs/](./docs/)