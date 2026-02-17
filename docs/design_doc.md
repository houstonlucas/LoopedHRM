# Two-Level Adaptive Compute — Hierarchical Iterative Solver (Sudoku First)

## Purpose

Build a system that learns **how much computation to use** while solving structured reasoning problems.

Development environment note: use `uv` for dependency management and execution
(`uv sync --extra dev`, then `uv run <command>`).

The system combines:

- **Iterative latent refinement** (LoopLM-style thinking in representation space)
- **Hierarchical compute control** (HRM-style multi-stage reasoning)

The focus is **adaptive computation**, not language modeling.

---

## Conceptual References

### Looped Latent Reasoning
- Paper: https://arxiv.org/pdf/2510.25741
- Adopted ideas:
  - repeated hidden-state refinement
  - learned halting for iterative computation

### Hierarchical Reasoning Model (HRM)
- Paper: https://arxiv.org/pdf/2506.21734
- Adopted ideas:
  - multiple reasoning time scales
  - staged compute with segment-level transitions

---

## Core Architecture Idea

The solver maintains an internal representation of the puzzle state and reasons via two interacting modules:

1. **Lower Module (Local Refinement)**
   - repeatedly updates latent state
   - performs local correction and propagation
   - uses learned halt to stop (or continue) refinement

2. **Upper Module (Global Strategy + Evaluation)**
   - performs global state updates and strategy shifts
   - evaluates candidate progress after each lower burst
   - decides whether to continue another cycle or terminate

### Pseudo-flow (Upper/Lower Back-and-Forth)

```text
Upper: global update / evaluate
Lower: refine
Lower: refine

Upper: update strategy / evaluate
Lower: refine
Lower: refine
Lower: refine

Upper: evaluate
Lower: refine

Upper: decide solution sufficient
Return solution
```

Key property: the number of consecutive `Lower: refine` steps is **variable** and learned, not fixed.

---

## Environment: Sudoku

### Task
- Input: partially-filled grid
- Output: completed valid grid
- Success: all Sudoku constraints satisfied

### Why Sudoku
- iterative reasoning is required
- correctness is clear and binary
- puzzle difficulty varies naturally
- compute usage can be measured directly

---

## Training Principles

Optimize for both:

- **Correctness:** solve puzzles accurately
- **Efficiency:** penalize unnecessary compute

Implementation details (loss forms, schedules, regularization) remain flexible while preserving these goals.

---

## Measurements

Track:

- solved puzzle rate
- lower-module refinement steps used
- upper-module cycles used
- performance vs compute tradeoff
- compute allocation across difficulty levels

---

## Desired Emergent Behavior

- easy puzzles should use little compute
- hard puzzles should use more compute
- lower vs upper compute should specialize
- strategy should vary with puzzle type/difficulty

---

## Scope Guardrails

The implementation should preserve the core ideas:

- latent iterative refinement
- hierarchical staged reasoning
- learned halting at both levels
- adaptive compute allocation

Architecture details are intentionally flexible as long as these invariants are retained.
