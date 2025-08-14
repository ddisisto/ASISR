# SPECTRA Development Guide

**Development Authority**: This document defines Claude Code optimization, workflow standards, and development best practices for the SPECTRA project throughout its entire lifecycle.

## Current Focus *(Update as project progresses)*

**Active Phase**: Phase 2B - Dynamic Spectral Strategies  
**Central Question**: *"Can training-phase-dependent spectral control optimize performance-variance trade-offs based on empirical Phase 2A findings?"*  
**Next Milestone**: Dynamic σ scheduling and training-phase control implementation  
**Status**: Phase 2A completed with multi-σ framework implemented and validated across datasets

**Immediate Priorities**:
1. Design Phase 2B dynamic spectral control experiments
2. Implement training-phase σ scheduling strategies
3. Test hybrid exploration-exploitation approaches
4. Validate dynamic strategies against best fixed σ approaches

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
When starting a new Claude Code context, use this streamlined succession protocol:

```bash
claude --prompt "continue at docs/phase2b-init.md, check 2-3 most recent git logs in minor detail and confirm all relevant context"
```

**How it works**:
1. Read the current phase initialization document (docs/phase2b-init.md)  
2. Review recent commits (`git log --oneline -3`) for immediate context
3. Validate current repository state and proceed with phase objectives
4. All context and next steps are captured in living phase documents

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
├── asisr/                                          # Main package (fully implemented)
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
python -c "from asisr.models.mlp import SpectralMLP; from asisr.regularization.spectral import AdaptiveSpectralRegularizer; print('SPECTRA package fully functional')"

# Run Phase 2A validation suite
python -m pytest tests/ -v --cov=asisr

# Check current git status
git status --porcelain
```

## Development Workflow & Standards

### **Phase-Based Development** *(MANDATORY)*

**Phase Discipline**: Never mix implementation across phases. Complete current phase validation before proceeding.

**Current Phase 2B Requirements**:
- Design dynamic σ scheduling algorithms (linear, exponential, step-wise)
- Implement training-phase-dependent spectral control
- Compare dynamic vs fixed σ strategies across datasets
- Validate hybrid exploration-exploitation approaches

**Validation Gate**: Phase 2B complete when dynamic strategies demonstrate superior or equivalent performance-variance trade-offs compared to best fixed σ approaches from Phase 2A.

### **Architecture Compliance** *(CRITICAL)*

**Single Source of Truth**: ARCHITECTURE.md defines all structural decisions  
**Plugin Architecture**: All new components must implement defined abstract interfaces
**No Architectural Drift**: Never deviate from ARCHITECTURE.md without updating it first

**Key Interfaces to Implement**:
- `SpectralRegularizedModel` for all neural network architectures
- `SpectralRegularizer` for all regularization methods  
- `CriticalityMonitor` for all assessment metrics

### **Code Quality Standards** *(NON-NEGOTIABLE)*

**Testing Requirements**:
- Unit tests for all modules in `asisr/` package
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
- Dataset files stored within package (`asisr/data/`)
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
- Literature research and analysis
- Focused module implementation (single responsibility)
- Visualization and figure generation
- Testing and validation workflows

**Parallel Development** (separate contexts):
- Environment setup and dependency management
- Performance optimization and profiling
- Documentation and tutorial creation
- Advanced theoretical development

## Common Development Patterns

### **Migration from Prototypes**

**Standard Migration Process**:
1. Read prototype code for logic understanding
2. Design interfaces per ARCHITECTURE.md patterns
3. Implement with proper abstractions and testing
4. Validate against original functionality
5. Remove prototype dependency

**Example**: `prototypes/map_loader.py` → `asisr/data/map_loader.py`
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
pytest tests/ -v --cov=asisr --cov-report=html

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
- `asisr/` - Main package (fully implemented with multi-σ framework)
- `prototypes/` - Legacy code (successfully migrated)
- `requirements.txt` - Dependencies (all installed and validated)

**Reference Materials**:
- `docs/START.md` - Original project conception
- `docs/2410.02536v3.txt` - Intelligence at Edge of Chaos paper
- `docs/DL-TRANSCRIPT.md` - Welch Labs geometric insights
- `asisr/data/Baarle-Nassau_-_Baarle-Hertog-en.svg` - Boundary map dataset

### **Import Patterns**

**Package Import Style**:
```python
# Core package imports
from asisr.models import SpectralMLP
from asisr.regularization import AdaptiveSpectralRegularizer
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
find asisr/ -name "*.py" -exec python -m py_compile {} \;

# Import chain testing
python -c "from asisr.data import map_loader; print('Data imports OK')"

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