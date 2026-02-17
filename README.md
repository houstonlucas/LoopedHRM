# LoopedHRM

A research codebase for **Two-Level Adaptive Compute — Hierarchical Iterative Solver** with Sudoku as the first environment.

This repository is intentionally scaffolded as a skeleton. It captures project structure, conceptual interfaces, and implementation placeholders while preserving the core design goals:

- iterative latent refinement (Looped-style)
- hierarchical reasoning stages (HRM-style)
- learned halting at both refinement and segment levels
- adaptive compute allocation based on puzzle difficulty

## Project Status

🚧 Early scaffold stage. Modules currently contain interfaces and TODO markers.

## Core Research Goal

Learn **how much computation to use** while solving structured reasoning tasks.

Two compute levels are modeled as interacting modules:

1. **Lower module** for local latent refinement (variable refine-step bursts)
2. **Upper module** for global update, evaluation, and continuation/stop decisions

The model should learn to stop both levels dynamically and allocate lower/upper compute based on puzzle difficulty.

## Repository Layout

```text
.
├── configs/                     # Experiment and model configuration skeletons
├── docs/
│   └── design_doc.md            # Living design document from the proposal
├── notebooks/                   # Analysis notebooks (future)
├── scripts/                     # Training/evaluation entrypoints
├── src/looped_hrm/
│   ├── data/                    # Sudoku generation/loading and batching
│   ├── models/                  # Hierarchical iterative solver components
│   ├── compute/                 # Halting and compute budget policies
│   ├── training/                # Losses, trainer loops, curriculum
│   ├── evaluation/              # Metrics and benchmarking
│   └── utils/                   # Common helpers
└── tests/                       # Unit/integration test skeletons
```

## Quick Start (Planned)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python scripts/train.py --config configs/experiment/sudoku_baseline.yaml
```

> The training pipeline is not yet implemented; current scripts are placeholders.

## Development Priorities

1. Implement Sudoku dataset and difficulty buckets.
2. Implement latent state update/refinement module.
3. Implement dual halting heads:
   - refinement halt
   - segment halt
4. Add training objective balancing correctness vs compute penalty.
5. Instrument compute-usage metrics by difficulty.

## Design Source

See `docs/design_doc.md` for the full conceptual plan and constraints.
