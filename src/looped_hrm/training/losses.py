"""Loss placeholders for correctness + compute efficiency objectives."""

from typing import Any


def correctness_loss(prediction: Any, target: Any) -> Any:
    """TODO: implement task correctness objective."""
    raise NotImplementedError


def compute_penalty(refine_steps: int, segments: int) -> float:
    """Simple placeholder compute penalty."""
    return float(refine_steps + segments)
