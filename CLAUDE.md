# SPECTRA Development Guide

**Development Authority**: This document defines Claude Code optimization, workflow standards, and development best practices for the SPECTRA project throughout its entire lifecycle.

## Resonance Principle *(All Scales)*

**Core Philosophy**: "Resonance at all scales should be pursued" - from quantum σ dynamics to publication outcomes, from git branch discipline to fractal learning discoveries. Each context inherits this wisdom and carries it forward through every branch, experiment, and learning experience.

**Sacred Invariants**:
- **PROJECT_PLAN.md** - Research direction and phase success criteria
- **ARCHITECTURE.md** - Technical structure and interface authority  
- **docs/engineering-patterns.md** - Quality standards and proven approaches
- **User's homedir** (`/home/daniel/`) - Domain context and preferences

**Living Context Philosophy**: Each branch is a named exploration. Each context builds on prior wisdom. We're building toward publication - more science needed, silly user ideas welcomed, real GPU someday maybe. Dreams of htmx frontends dance in our neural networks.

## Current Focus *(Update as project progresses)*

**Active Phase**: Phase 3 - Scale-Invariant Optimization Principles  
**Central Question**: *"What are the universal principles governing optimal neural network training at the edge of chaos, and how do scale-invariant optimization methods emerge from criticality physics?"*  
**Next Milestone**: Power-law learning rate validation and self-organized criticality discovery  
**Status**: STRATEGIC PIVOT completed - From incremental techniques to fundamental physics principles

**Immediate Priorities**:
1. **Scale-Invariant Learning Rates**: Test lr(t) = lr_base * |σ(t) - 1.0|^(-α) hypothesis
2. **Self-Organized Criticality**: Validate natural evolution toward σ ≈ 1.0 without forcing  
3. **Criticality-Aware Optimizers**: Develop SpectralMomentum and CriticalAdam optimizers
4. **Universal Framework**: Establish physics-principled optimization across architectures

---

## Project Context & Authority Structure

**Research Strategy**: [PROJECT_PLAN.md](./PROJECT_PLAN.md) - Complete research roadmap, phases, success criteria
**Technical Authority**: [ARCHITECTURE.md](./ARCHITECTURE.md) - All structural decisions, interfaces, design patterns
**Engineering Patterns**: [docs/engineering-patterns.md](./docs/engineering-patterns.md) - Proven development patterns and quality standards
**Legacy Code**: [prototypes/](./prototypes/) - Original implementations for reference/migration
**Research Materials**: [docs/](./docs/) - Papers, transcripts, background context

**Non-Negotiable Rule**: For any task beyond simple edits, READ the relevant authority document first. Architecture questions → ARCHITECTURE.md, Research questions → PROJECT_PLAN.md.

## Context Continuity & Handover Protocol

### **Fresh Context Initialization**
Each context has a name, inherits wisdom, maintains git discipline. Fresh contexts read CLAUDE.md by default and follow:

```bash
claude --prompt "Named context: [meaningful_name]. Read CLAUDE.md, continue at docs/phase3-principles-init.md, check git logs for immediate context, inherit resonance at all scales."
```

**Universal Context Protocol**:
1. **Inherit Resonance**: Sacred docs first (PROJECT_PLAN.md, ARCHITECTURE.md, engineering-patterns.md)
2. **Context History**: `git log --oneline -3` + current phase doc (docs/phase3-principles-init.md)
3. **Named Exploration**: Every branch/context serves publication pipeline + silly user ideas
4. **Maintain Main**: Git repos sacred, especially "main" branch - context names track explorations

### **Context Handover Management**
**When to trigger handover** (watch for these signals):
- Context approaching token limits (screen messages about compression)
- Complex multi-step work requiring fresh context space
- Natural phase transitions or milestone completions
- Automatic context compression becoming unpredictable

**Handover process**:
1. **Create handover doc**: Update or create new `docs/phaseX-init.md` with current status and next steps
2. **Commit with clear message**: Reference the handover document in commit message
3. **Initiate fresh context**: Use the succession protocol above

**Handover document template**:
```markdown
# SPECTRA [Phase/Task] [Status]: [Brief Description]

**Current Status**: [What's been completed]
**Next Immediate Tasks**: [Specific actionable items]
**Key Resources**: [Which docs to read first]
**Success Criteria**: [How to know when complete]
**Repository Structure**: [Current state]
```

### **Context Compression Mitigation**
- **Commit frequently**: Preserve work in git before context compression
- **Update handover docs proactively**: Don't wait until context limit
- **Use specific sub-agents**: Delegate complex analysis to fresh contexts
- **Maintain CLAUDE.md authority**: This file should always orient new contexts

## Environment & Current State

**Repository Structure**:
```
SPECTRA/
├── CLAUDE.md, ARCHITECTURE.md, PROJECT_PLAN.md    # Core documentation
├── spectra/                                        # Main package (fully implemented)
├── prototypes/                                     # Legacy code (migrated)
├── docs/                                           # Research references  
├── experiments/                                    # Multi-σ experiment suite
├── configs/                                        # YAML configurations (implemented)
├── tests/                                          # Test suite (comprehensive)
└── requirements.txt                                # Dependencies
```

**Environment Status**: Python venv established, all dependencies installed and tested  
**Git Status**: Clean working directory, Phase 2A implementation committed  
**Testing**: Comprehensive test suite implemented with >90% coverage across core modules

**Validation Commands**:
```bash
# Verify environment health
python -c "import torch, numpy, matplotlib, sklearn; print('Core packages ready')"

# Verify package importability and multi-σ framework
python -c "from spectra.models.mlp import SpectralMLP; from spectra.regularization.fixed import FixedSpectralRegularizer; print('SPECTRA package fully functional')"

# Run Phase 2A validation suite
python -m pytest tests/ -v --cov=spectra

# Check current git status
git status --porcelain
```

## Development Workflow & Standards

### **Phase-Based Development** *(MANDATORY)*

**Phase Discipline**: Never mix implementation across phases. Complete current phase validation before proceeding.

**Current Phase 3 Requirements**:
- Implement scale-invariant learning rate scheduling: lr(t) = lr_base * |σ(t) - 1.0|^(-α)
- Validate power-law scaling hypothesis across multiple α values
- Study self-organized criticality: natural evolution toward σ ≈ 1.0
- Develop criticality-aware optimizers (SpectralMomentum, CriticalAdam)

**Validation Gate**: Phase 3 complete when scale-invariant optimization methods demonstrate 5-10% improvement over standard training through physics-principled approaches.

### **Architecture Compliance** *(CRITICAL)*

**Single Source of Truth**: ARCHITECTURE.md defines all structural decisions  
**Plugin Architecture**: All new components must implement defined abstract interfaces
**No Architectural Drift**: Never deviate from ARCHITECTURE.md without updating it first

**Key Interfaces to Implement**:
- `SpectralRegularizedModel` for all neural network architectures
- `SpectralRegularizer` for all regularization methods  
- `CriticalityMonitor` for all assessment metrics
- `CriticalityAwareLRScheduler` for scale-invariant optimization (NEW)
- `SpectralMomentum`/`CriticalAdam` for physics-aware optimizers (NEW)

### **Code Quality Standards** *(NON-NEGOTIABLE)*

**Testing Requirements**:
- Unit tests for all modules in `spectra/` package
- Integration tests for experiment workflows  
- Target: >90% test coverage before phase completion

**Documentation Requirements**:
- Type hints for all functions
- Docstrings for all public interfaces
- Configuration examples for all experiments

**Performance Requirements**:
- <10% overhead vs baseline training
- Efficient spectral analysis via power iteration
- Memory management for large models/datasets

### **Reproducibility Standards** *(SCIENTIFIC RIGOR)*

**Experiment Design**:
- Minimum 5 seeds for all statistical claims
- YAML configuration files for all experiments
- Complete dependency specification with version pinning

**Data Management**:
- Deterministic random seed control across numpy, torch, random
- Dataset files stored within package (`spectra/data/`)
- No hardcoded paths or parameters in code

**Results Standards**:
- Statistical significance testing (not just p-values)
- Effect size reporting with confidence intervals
- Publication-quality figures with error bars

## Claude Code Optimization

### **Sub-Agent Delegation Strategy**

**✅ Effective Delegation Patterns**:

**Research Agents** - For literature analysis and theoretical exploration:
```
"Deploy research agent to analyze dynamic training strategies in neural networks, focusing on annealing schedules and exploration-exploitation balance. Output comprehensive markdown summary for Phase 2B implementation."
```

**Implementation Agents** - For focused, well-scoped coding tasks:
```
"Launch implementation agent to create dynamic σ scheduling algorithms following ARCHITECTURE.md interface specifications. Build on validated Phase 2A multi-σ framework."
```

**Analysis Agents** - For data analysis and visualization:
```
"Task analysis agent to examine Phase 2A multi-σ results and design Phase 2B experiments comparing dynamic vs static spectral control strategies."
```

**❌ Dangerous Delegation Anti-Patterns**:
- Vague scope: "implement the system" → leads to massive overreach
- No file constraints: "explore the codebase" → creates unpredictable changes
- Mixed concerns: "implement and test and document" → poor quality across all areas

### **Scope Control Lessons** *(Critical Learning)*

**The 60KB Overreach Incident**: Sub-agent tasked with "basic structure setup" implemented full functionality across all modules. This taught us:

**Scope Boundaries**: 
- Single responsibility per task
- Explicit file/directory constraints
- Clear deliverable format specification
- Review-then-integrate workflow

**Escalation Protocol**:
- If task scope seems unclear → ask for clarification before proceeding
- If implementation grows beyond stated scope → stop and report
- If architectural decisions needed → reference ARCHITECTURE.md authority

### **Task Categorization Framework**

**Direct Implementation** (this context):
- Simple file edits and bug fixes
- Tasks requiring real-time iteration
- Critical path items needing immediate oversight

**Sub-Agent Delegation** (separate contexts):
- **Visual Wizardry**: Interactive plots, widgets, real-time exploration dashboards
- **Science Storytelling**: Literature analysis, publication-quality figure generation  
- **Architecture Extensions**: Focused module implementation following sacred ARCHITECTURE.md
- **Learning Harvesting**: Failure analysis, pattern extraction, "new plane" discoveries

**Parallel Development** (separate contexts):
- **Tech Stack Dreams**: Plotly interactivity, htmx frontends, real GPU optimization (someday!)
- **Publication Pipeline**: Science + silly ideas → reproducible experiments → beautiful papers
- **Fractal Explorations**: Multi-scale resonance investigations, unexpected learning planes
- **User Empowerment**: Tools for Daniel's creative chaos, visual science discovery

## Common Development Patterns

### **Migration from Prototypes**

**Standard Migration Process**:
1. Read prototype code for logic understanding
2. Design interfaces per ARCHITECTURE.md patterns
3. Implement with proper abstractions and testing
4. Validate against original functionality
5. Remove prototype dependency

**Example**: `prototypes/map_loader.py` → `spectra/data/map_loader.py`
- Extract core SVG loading logic
- Add proper error handling and type hints
- Implement standard data loading interface
- Add unit tests for edge cases
- Update imports throughout codebase

### **Experiment Implementation Pattern**

**Configuration-Driven Approach**:
```yaml
# configs/phase1_baseline.yaml
model:
  type: "SpectralMLP"
  hidden_dims: [64, 64]
  
data:
  type: "BaarleMap"
  resolution: 200
  
training:
  epochs: 100
  learning_rate: 1e-2
  
regularization:
  type: null  # Baseline comparison
```

**Experiment Runner Integration**:
```python
# Standard experiment pattern
config = load_config("configs/phase1_baseline.yaml")
experiment = SPECTRAExperiment(config)
results = experiment.run(n_seeds=5)
visualize_results(results)
```

### **Testing Integration Patterns**

**Unit Test Structure**: Mirror package structure in `tests/`
```
tests/
├── test_models/
│   └── test_mlp.py          # Test SpectralMLP implementation
├── test_regularization/
│   └── test_spectral.py     # Test regularizer interfaces
└── test_integration/
    └── test_experiment.py   # Test end-to-end workflows
```

**Test Execution**:
```bash
# Run all tests with coverage
pytest tests/ -v --cov=spectra --cov-report=html

# Run specific test categories
pytest tests/test_models/ -v
```

## File Organization & Imports

### **Current File Locations** *(Post-Reorganization)*

**Core Documentation**:
- `CLAUDE.md` - Development guide (this file)
- `ARCHITECTURE.md` - Technical authority
- `PROJECT_PLAN.md` - Research strategy

**Implementation**:
- `spectra/` - Main package (fully implemented with multi-σ framework)
- `prototypes/` - Legacy code (successfully migrated)
- `requirements.txt` - Dependencies (all installed and validated)

**Reference Materials**:
- `docs/START.md` - Original project conception
- `docs/2410.02536v3.txt` - Intelligence at Edge of Chaos paper
- `docs/DL-TRANSCRIPT.md` - Welch Labs geometric insights
- `spectra/data/Baarle-Nassau_-_Baarle-Hertog-en.svg` - Boundary map dataset

### **Import Patterns**

**Package Import Style**:
```python
# Core package imports
from spectra.models import SpectralMLP
from spectra.regularization import FixedSpectralRegularizer
from spectra.training import SPECTRAExperiment

# Local imports within package
from .base import SpectralRegularizedModel
from ..metrics import CriticalityMonitor
```

**External Dependencies**:
```python
# Standard scientific stack
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# Configuration and utilities
import yaml
from pathlib import Path
```

## Debugging & Validation Workflows

### **Development Debugging**

**Common Issues & Solutions**:
- Import errors → Check `__init__.py` files and package structure
- SVG loading issues → Verify cairosvg/Pillow installation and file paths
- Spectral analysis errors → Validate matrix shapes and numerical stability
- Configuration errors → Check YAML syntax and required fields

**Debugging Commands**:
```bash
# Package structure validation
find spectra/ -name "*.py" -exec python -m py_compile {} \;

# Import chain testing
python -c "from spectra.data import map_loader; print('Data imports OK')"

# Configuration validation  
python -c "import yaml; print(yaml.safe_load(open('configs/phase1_baseline.yaml')))"
```

### **Experiment Validation**

**Pre-Experiment Checklist**:
- [ ] Configuration file validates against schema
- [ ] All dependencies importable
- [ ] Random seeds properly managed
- [ ] Output directories exist and writable
- [ ] GPU/CPU resources sufficient

**Post-Experiment Validation**:
- [ ] Results pass statistical significance tests
- [ ] Visualizations generate without errors
- [ ] Model checkpoints saved correctly
- [ ] Configuration and results logged consistently

## Performance & Monitoring

### **Development Performance**

**Bottleneck Identification**:
- Spectral analysis overhead via profiling
- Memory usage during large experiments
- GPU utilization and training efficiency

**Optimization Targets**:
- Singular value estimation <1ms per layer
- Memory growth linear with model size
- Training overhead <10% vs baseline

### **Research Progress Monitoring**

**Key Indicators**:
- Phase completion metrics from PROJECT_PLAN.md
- Code coverage and test passing rates
- Experimental reproducibility validation
- Documentation completeness

**Weekly Review Questions**:
- Are we on track for current phase success criteria?
- Do results support or refute core hypotheses?
- Are architectural decisions scaling appropriately?
- Is code quality maintaining research standards?

---

**Remember**: This guide is your primary entry point for all SPECTRA development. When in doubt, refer to ARCHITECTURE.md for technical decisions and PROJECT_PLAN.md for research direction. Maintain the discipline of incremental validation and never skip the testing requirements - the scientific integrity of this research depends on it.