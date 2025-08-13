# ASISR Project - Claude Code Development Guide

## Project Overview

This is the **Adaptive Scale-Invariant Spectral Regularization (ASISR)** project, focused on neural network boundary learning experiments using the Belgium-Netherlands border map as a complex real-world dataset. The project explores spectral regularization techniques to achieve optimal training dynamics at the "edge of chaos."

**Architecture Authority**: All structural decisions are documented in [ARCHITECTURE.md](./ARCHITECTURE.md) - this is the single source of truth for code organization, interfaces, and design patterns.

## Environment Setup

**Virtual Environment**: Always work within the activated virtual environment (`venv/`)

**Dependencies**: 
- Install via: `pip install -r requirements.txt`
- Core packages: torch, numpy, scikit-learn, matplotlib, cairosvg, Pillow
- All packages verified working in Python 3.13.3

**Test Environment Health**: 
```bash
python -c "import numpy, torch, matplotlib, sklearn, cairosvg; from PIL import Image; print('All critical packages imported successfully')"
```

## Core Development Principles

### 1. Architecture Compliance (CRITICAL)
- **Follow ARCHITECTURE.md religiously** - never deviate without updating the architecture document first
- Implement the plugin interfaces: `SpectralRegularizedModel` and `SpectralRegularizer` base classes
- Maintain the modular package structure: `asisr/{models,regularization,metrics,data,training,visualization,utils}/`
- All new components must be independently usable and testable

### 2. Phase-Based Development (MANDATORY)
- **Phase 1**: Boundary mapping proof-of-concept (current focus)
- **Phase 2**: Adaptive ASISR implementation  
- **Phase 3**: Multi-scale architecture
- **NEVER skip phases or mix implementation across phases**
- Each phase must validate completely before proceeding
- Experiment organization mirrors `/experiments/phase{N}_*/` structure

### 3. Metrics-Driven Development
- All decisions guided by criticality indicators:
  - Dead neuron rate (threshold: 1e-5)
  - Perturbation sensitivity (eps: 1e-3) 
  - Fractal dimension of decision boundaries
  - Spectral properties (singular value distributions)
- Implement comprehensive logging of all metrics
- Statistical analysis requires minimum 5-seed runs

### 4. Reproducibility First
- **All experiments must be deterministic with seed control**
- Use configuration files (YAML) for all experimental parameters
- Never hardcode paths, values, or experimental settings
- Implement proper random seed management across numpy, torch, random

## Code Quality Standards

### Mandatory Requirements
- **Test Coverage**: >90% for all core modules in `asisr/`
- **Type Hints**: All functions must have complete type annotations
- **Docstrings**: All public interfaces require comprehensive documentation
- **Performance**: <10% overhead vs baseline training
- **Interfaces**: All new regularization/model classes implement base interfaces

### Code Organization
- Core library: `asisr/` package with proper `__init__.py` files
- Experiments: Phase-specific directories under `experiments/`
- Configuration: YAML files in `configs/` directory
- Tests: Mirror package structure in `tests/`
- Scripts: Utility scripts in `scripts/`

### Integration with Existing Code
Current files integration strategy:
- `map_loader.py` → `asisr/data/map_loader.py`
- `SAMPLE-CODE-v1.md` logic → `asisr/models/mlp.py` + `asisr/training/experiment.py`
- Existing metrics functions → `asisr/metrics/criticality.py`

## Sub-Agent Delegation Strategy

### When to Use Sub-Agents

**✅ Research Agents** - For complex analysis and exploration:
```
"Deploy research agent to analyze all spectral regularization literature and identify key mathematical relationships for implementation"
```

**✅ Implementation Agents** - For focused coding tasks:
```
"Launch implementation agent to create the AdaptiveSpectralRegularizer class following the interface specifications in ARCHITECTURE.md, exploring existing regularization patterns"
```

**✅ Analysis Agents** - For data analysis and visualization:
```
"Use analysis agent to examine Phase 1 boundary mapping results and generate publication-quality comparison plots between baseline and spectral-regularized approaches"
```

**✅ Architecture Agents** - For structural decisions:
```
"Task architecture agent to review current codebase and propose optimal integration strategy for transformer spectral regularization"
```

### Sub-Agent Guidelines
- **Clear Scope**: Specify exactly which files/directories to examine
- **Precise Instructions**: Define the expected deliverable format
- **Context Boundaries**: Limit exploration to relevant project areas  
- **Output Format**: Single MD file, specific code module, analysis report, etc.
- **Handoff Protocol**: Sub-agent outputs become authoritative context

### When NOT to Use Sub-Agents
- ❌ Simple file edits or minor bug fixes
- ❌ Tasks requiring real-time iteration or debugging
- ❌ When maintaining conversation context is critical
- ❌ Quick verification or testing tasks

## Key File Locations

**Architecture & Planning**:
- `ARCHITECTURE.md` - System design authority
- `PROJECT_PLAN.md` - Development roadmap
- `requirements.txt` - Dependency specification

**Core Implementation**:
- `map_loader.py` - Belgium-Netherlands boundary data loader  
- `example_usage.py` - Integration examples
- `SAMPLE-CODE-v1.md` - Original experiment code (migration source)

**Documentation**:
- `MAP_LOADER_README.md` - Data loader documentation
- `DL-TRANSCRIPT.md` - Research context
- `START.md` - Project initiation guide

## Current Phase 1 Priorities

1. **Environment Validation** ✅ (completed)
2. **Code Architecture Migration** (next priority)
   - Extract MLP model to `asisr/models/mlp.py`
   - Extract metrics to `asisr/metrics/criticality.py`  
   - Extract experiment runner to `asisr/training/experiment.py`
3. **Baseline Comparison Implementation**
4. **Belgium-Netherlands Boundary Experiments**
5. **Phase 1 Validation and Documentation**

## Testing Strategy

- **Unit Tests**: Each module in `asisr/` package
- **Integration Tests**: End-to-end experiment workflows
- **Validation Tests**: Scientific reproducibility verification
- **Performance Tests**: Overhead measurement vs baseline

Run tests: `pytest tests/ -v --cov=asisr --cov-report=html`

## Experiment Workflow

1. **Configuration**: Define experiment in `configs/{experiment}.yaml`
2. **Implementation**: Use `ASISRExperiment` orchestration class
3. **Execution**: Multi-seed statistical runs (minimum 5 seeds)
4. **Analysis**: Automated metric collection and visualization
5. **Documentation**: Results integration into project documentation

## Common Commands

```bash
# Environment activation (if needed)
source venv/bin/activate

# Dependency installation
pip install -r requirements.txt

# Test imports
python -c "import asisr; print('ASISR package ready')"

# Run experiments (future)
python scripts/run_experiments.py --config configs/boundary_mapping.yaml

# Generate figures (future)
python scripts/generate_figures.py --experiment results/phase1_baseline/
```

---

**Remember**: ARCHITECTURE.md is the single source of truth for all structural decisions. This guide focuses on development workflow, coding standards, and architectural compliance enforcement. Always validate changes against the architecture document and update it when making structural modifications.