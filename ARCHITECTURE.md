# ASISR Code Architecture

## Executive Summary

This document presents the optimal code architecture for the Adaptive Scale-Invariant Spectral Regularization (ASISR) project. The architecture balances research flexibility with engineering rigor, supporting both the current boundary mapping proof-of-concept and future extensions to transformers and other architectures.

## Proposed Directory Structure

```
ASISR/
├── asisr/                           # Core library package
│   ├── __init__.py
│   ├── models/                      # Neural network architectures
│   │   ├── __init__.py
│   │   ├── mlp.py                   # Enhanced MLP with spectral hooks
│   │   ├── transformer.py           # Future: Transformer with attention spectral regularization
│   │   └── base.py                  # Abstract base classes and interfaces
│   ├── regularization/              # Spectral regularization components
│   │   ├── __init__.py
│   │   ├── spectral.py              # Core spectral regularization methods
│   │   ├── adaptive.py              # AdaptiveSpectralRegularizer
│   │   ├── multi_scale.py           # Multi-layer hierarchical regularization
│   │   └── utils.py                 # Singular value estimation, power iteration
│   ├── metrics/                     # Criticality and performance metrics
│   │   ├── __init__.py
│   │   ├── criticality.py           # Dead neurons, sensitivity, fractal dimension
│   │   ├── boundary.py              # Decision boundary analysis
│   │   ├── fractal.py               # Multi-scale fractal analysis
│   │   └── spectral_metrics.py      # Spectral radius tracking, power law analysis
│   ├── data/                        # Data loading and preprocessing
│   │   ├── __init__.py
│   │   ├── map_loader.py            # Belgium-Netherlands boundary loader (existing)
│   │   ├── synthetic.py             # Two-moons and other synthetic datasets
│   │   └── transforms.py            # Data augmentation and coordinate transforms
│   ├── training/                    # Training orchestration
│   │   ├── __init__.py
│   │   ├── experiment.py            # Experiment runner with multi-seed support
│   │   ├── optimization.py          # Scale-invariant optimizers and schedulers
│   │   └── hooks.py                 # Training hooks for real-time monitoring
│   ├── visualization/               # Plotting and analysis tools
│   │   ├── __init__.py
│   │   ├── boundaries.py            # Decision boundary visualization
│   │   ├── dynamics.py              # Training dynamics plots
│   │   ├── criticality.py           # Criticality indicator plots
│   │   └── comparative.py           # Multi-experiment comparisons
│   └── utils/                       # General utilities
│       ├── __init__.py
│       ├── seed.py                  # Reproducibility utilities
│       ├── device.py                # GPU/CPU management
│       └── config.py                # Configuration management
├── experiments/                     # Experiment scripts and notebooks
│   ├── __init__.py
│   ├── phase1_boundary_mapping/     # Phase 1: Boundary mapping proof-of-concept
│   │   ├── baseline_comparison.py
│   │   ├── baarle_experiment.py
│   │   └── analysis.ipynb
│   ├── phase2_adaptive_asisr/       # Phase 2: Adaptive spectral regularization
│   │   ├── adaptive_targeting.py
│   │   ├── criticality_monitoring.py
│   │   └── parameter_sweeps.py
│   ├── phase3_multi_scale/          # Phase 3: Multi-scale architecture
│   │   ├── hierarchical_regularization.py
│   │   ├── layer_analysis.py
│   │   └── scaling_laws.py
│   └── notebooks/                   # Analysis and visualization notebooks
│       ├── boundary_analysis.ipynb
│       ├── criticality_dashboard.ipynb
│       └── results_comparison.ipynb
├── tests/                          # Unit tests and integration tests
│   ├── __init__.py
│   ├── test_models/
│   ├── test_regularization/
│   ├── test_metrics/
│   └── test_integration/
├── scripts/                        # Utility scripts
│   ├── setup_environment.py
│   ├── run_experiments.py
│   └── generate_figures.py
├── configs/                        # Configuration files
│   ├── base_config.yaml
│   ├── boundary_mapping.yaml
│   └── adaptive_asisr.yaml
├── requirements.txt
├── setup.py
└── README.md
```

### Rationale for Structure

1. **Modular Design**: Clear separation between models, regularization, metrics, and training allows independent development and testing
2. **Research Phases**: Experiment directory mirrors the project phases, enabling structured progression
3. **Reusability**: Core components in `asisr/` package can be imported and reused across different experiments
4. **Extensibility**: Plugin-like architecture for new regularization methods, metrics, and models
5. **Maintainability**: Standard Python package structure with proper testing and configuration management

## Core Classes and Interfaces

### 1. Base Abstractions

```python
# asisr/models/base.py
class SpectralRegularizedModel(nn.Module):
    """Abstract base class for models supporting spectral regularization"""
    
    def get_spectral_weights(self) -> List[torch.Tensor]:
        """Return weight matrices subject to spectral regularization"""
        raise NotImplementedError
    
    def spectral_regularization_loss(self, regularizer) -> torch.Tensor:
        """Compute spectral regularization loss"""
        raise NotImplementedError

# asisr/regularization/base.py  
class SpectralRegularizer:
    """Abstract base class for spectral regularization methods"""
    
    def compute_loss(self, model: SpectralRegularizedModel) -> torch.Tensor:
        """Compute regularization loss"""
        raise NotImplementedError
```

### 2. Core Regularization Components

```python
# asisr/regularization/adaptive.py
class AdaptiveSpectralRegularizer:
    """
    Adaptive spectral regularization with real-time criticality monitoring.
    
    Dynamically adjusts target sigma values based on criticality indicators:
    - Dead neuron rate
    - Perturbation sensitivity  
    - Fractal dimension of decision boundaries
    """
    
    def __init__(self, adaptation_rate=0.01, target_criticality=0.7):
        self.adaptation_rate = adaptation_rate
        self.target_criticality = target_criticality
        self.layer_targets = {}  # Per-layer sigma targets
    
    def update_targets(self, model, criticality_metrics) -> float:
        """Update sigma targets based on criticality assessment"""
        
    def compute_loss(self, model) -> torch.Tensor:
        """Compute adaptive spectral regularization loss"""

# asisr/regularization/multi_scale.py
class MultiScaleSpectralRegularizer:
    """
    Hierarchical spectral regularization inspired by renormalization group theory.
    Different layers have different optimal spectral properties.
    """
    
    def __init__(self, depth_scaling_factor=0.9):
        self.depth_scaling_factor = depth_scaling_factor
    
    def compute_layer_targets(self, model) -> Dict[int, float]:
        """Compute layer-specific sigma targets"""
```

### 3. Criticality Metrics System

```python
# asisr/metrics/criticality.py
class CriticalityMonitor:
    """
    Comprehensive criticality assessment combining multiple indicators.
    
    Metrics:
    - Dead neuron rate (low activation threshold)
    - Perturbation sensitivity (input perturbation response)
    - Fractal dimension (decision boundary complexity)
    - Spectral properties (singular value distributions)
    """
    
    def __init__(self, dead_threshold=1e-5, perturbation_eps=1e-3):
        self.dead_threshold = dead_threshold
        self.perturbation_eps = perturbation_eps
    
    def compute_criticality_score(self, model, data) -> Dict[str, float]:
        """Compute comprehensive criticality assessment"""
    
    def is_at_criticality(self, metrics: Dict[str, float]) -> bool:
        """Determine if model is operating at edge of chaos"""
```

### 4. Enhanced Model Architecture

```python
# asisr/models/mlp.py  
class SpectralMLP(SpectralRegularizedModel):
    """
    Enhanced MLP with spectral regularization support.
    
    Features:
    - Spectral norm tracking for each layer
    - Pre-activation recording for criticality analysis
    - Configurable spectral regularization hooks
    """
    
    def __init__(self, input_dim=2, hidden_dims=[64, 64], output_dim=1):
        super().__init__()
        self.build_layers(input_dim, hidden_dims, output_dim)
        self.spectral_hooks = []
        
    def forward_with_preactivations(self, x):
        """Forward pass recording pre-activations for analysis"""
        
    def get_spectral_weights(self) -> List[torch.Tensor]:
        """Return weight matrices for spectral analysis"""
```

### 5. Experiment Orchestration

```python
# asisr/training/experiment.py
class ASISRExperiment:
    """
    Comprehensive experiment runner for ASISR research.
    
    Features:
    - Multi-seed statistical analysis
    - Real-time metric tracking
    - Adaptive regularization scheduling
    - Visualization generation
    - Reproducibility guarantees
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.results = []
        self.setup_logging()
        
    def run(self, n_seeds=5) -> ExperimentResults:
        """Run complete experiment with statistical analysis"""
        
    def compare_methods(self, methods: List[str]) -> ComparisonResults:
        """Compare different regularization approaches"""
```

## Key Design Decisions

### 1. Plugin Architecture for Regularization
**Decision**: Regularization methods implement common interface  
**Rationale**: Easy to add new spectral regularization variants, compare methods systematically  
**Trade-off**: Slight complexity overhead vs. maximum flexibility for research

### 2. Metrics-Driven Adaptive System
**Decision**: Criticality monitoring drives adaptive regularization  
**Rationale**: Enables real-time optimization toward edge of chaos  
**Trade-off**: Computational overhead vs. superior training dynamics

### 3. Hierarchical Configuration System  
**Decision**: YAML configs for experiments, Python classes for core logic  
**Rationale**: Reproducible experiments, easy parameter sweeps, version control  
**Trade-off**: Additional configuration layer vs. experimental rigor

### 4. Modular Visualization System
**Decision**: Separate visualization package with specialized plotting functions  
**Rationale**: Research requires rich visualizations, but training code stays clean  
**Trade-off**: Code organization complexity vs. visualization quality

### 5. Phase-Based Development Structure
**Decision**: Experiment directories mirror project phases  
**Rationale**: Matches natural research progression, enables incremental validation  
**Trade-off**: Directory proliferation vs. clear development roadmap

## Integration Strategy with Existing Code

### Phase 1: Minimal Disruption Migration
```python
# Current SAMPLE-CODE-v1.md code can be migrated incrementally:

# 1. Extract existing MLP -> asisr/models/mlp.py (enhanced)
# 2. Extract metrics functions -> asisr/metrics/criticality.py  
# 3. Extract experiment runner -> asisr/training/experiment.py
# 4. Integrate map_loader.py -> asisr/data/map_loader.py
# 5. Create baseline experiment script using existing logic
```

### Phase 2: Enhanced Capabilities
```python
# Add adaptive spectral regularization:
# - Replace fixed target_sigma=1.0 with AdaptiveSpectralRegularizer
# - Add real-time criticality monitoring
# - Implement scale-invariant learning rate scheduling
```

### Phase 3: Advanced Features  
```python
# Multi-scale and transformer extensions:
# - Add MultiScaleSpectralRegularizer
# - Implement transformer attention matrix regularization
# - Add hierarchical fractal analysis
```

## Extension Points for Future Phases

### 1. New Architectures
- **Interface**: `SpectralRegularizedModel` base class
- **Extensions**: Transformers, CNNs, Graph Neural Networks
- **Requirement**: Implement `get_spectral_weights()` method

### 2. Regularization Methods
- **Interface**: `SpectralRegularizer` base class  
- **Extensions**: Layer-wise targeting, temporal dynamics, RG-inspired scaling
- **Requirement**: Implement `compute_loss()` method

### 3. Criticality Metrics
- **Interface**: Plugin system in `CriticalityMonitor`
- **Extensions**: New fractal measures, temporal criticality, network topology metrics
- **Requirement**: Return normalized criticality score [0, 1]

### 4. Optimization Methods
- **Interface**: PyTorch optimizer compatibility
- **Extensions**: Scale-invariant schedulers, critical-aware optimizers
- **Requirement**: Support spectral regularization loss terms

### 5. Visualization Capabilities
- **Interface**: Standardized plotting functions with consistent styling
- **Extensions**: Interactive dashboards, animation sequences, comparative analysis
- **Requirement**: Accept experiment results objects

## Dependencies and Tools

### Core Dependencies
```python
# requirements.txt (essential)
torch>=1.10.0
numpy>=1.20.0
scipy>=1.7.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pyyaml>=6.0
tqdm>=4.60.0
```

### Visualization Dependencies  
```python
# Enhanced plotting and analysis
seaborn>=0.11.0
plotly>=5.0.0  # Interactive plots
jupyter>=1.0.0  # Notebook support
cairosvg>=2.5.0  # SVG map rendering
pillow>=8.0.0  # Image processing
```

### Development Dependencies
```python
# Testing and code quality
pytest>=6.0.0
pytest-cov>=2.12.0
black>=21.0.0  # Code formatting
flake8>=3.9.0  # Linting
mypy>=0.910  # Type checking
```

### Research Extensions
```python
# Advanced analysis capabilities  
tensorboard>=2.7.0  # Training visualization
wandb>=0.12.0  # Experiment tracking (optional)
networkx>=2.6.0  # Graph analysis (for network topology)
numba>=0.54.0  # Performance optimization (optional)
```

## Implementation Roadmap

### Week 1-2: Foundation Migration
1. Create package structure and base classes
2. Migrate existing code to modular architecture
3. Implement basic experiment orchestration
4. Add comprehensive testing framework

### Week 3-4: Adaptive Regularization  
1. Implement `AdaptiveSpectralRegularizer`
2. Add real-time criticality monitoring
3. Create scale-invariant learning rate scheduling
4. Validate on Belgium-Netherlands boundary mapping

### Week 5-6: Multi-Scale Architecture
1. Implement `MultiScaleSpectralRegularizer` 
2. Add hierarchical fractal analysis
3. Create layer-wise spectral analysis tools
4. Validate scaling relationships

### Week 7-8: Documentation and Validation
1. Complete API documentation
2. Create tutorial notebooks
3. Implement comprehensive benchmarks
4. Prepare publication-ready results

## Success Metrics

### Code Quality
- **Test Coverage**: >90% for core modules
- **Documentation**: Complete API docs with examples
- **Performance**: <10% overhead vs. baseline training
- **Modularity**: Each component usable independently

### Research Capability
- **Reproducibility**: All experiments deterministic with seed control
- **Extensibility**: New regularization methods addable in <100 lines
- **Scalability**: Support for transformer-scale models
- **Visualization**: Publication-quality figures generated automatically

### Scientific Impact
- **Validation**: ASISR outperforms fixed spectral regularization
- **Generalization**: Framework works across multiple architectures
- **Insights**: Clear relationship between criticality and performance
- **Innovation**: Novel scale-invariant optimization methods discovered

---

This architecture provides a solid foundation for the ASISR project while maintaining the flexibility needed for cutting-edge research. The modular design enables rapid experimentation while the structured approach ensures reproducible, publication-quality results.