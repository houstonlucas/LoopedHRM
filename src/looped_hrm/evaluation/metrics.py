"""Evaluation metrics for correctness and compute usage."""

from typing import Dict


def compute_efficiency(solved_rate: float, avg_compute: float) -> float:
    """Placeholder efficiency score for early experimentation."""
    if avg_compute <= 0:
        return 0.0
    return solved_rate / avg_compute


def summarize_metrics(metrics: Dict[str, float]) -> Dict[str, float]:
    """TODO: add richer aggregate reporting by difficulty bucket."""
    return metrics
