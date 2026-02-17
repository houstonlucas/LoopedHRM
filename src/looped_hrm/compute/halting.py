"""Halting policies for refinement and segment-level adaptive compute."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _state_to_vector(state: Any) -> torch.Tensor:
    arr = torch.as_tensor(state, dtype=torch.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    reduction_axes = tuple(range(arr.ndim - 1))
    return arr.mean(dim=reduction_axes)


class _SigmoidHaltingPolicy(nn.Module):
    """Shared logistic continue/stop logic used by both halting policies."""

    def __init__(
        self,
        min_steps: int,
        max_steps: int,
        continue_threshold: float,
        seed: int,
        init_scale: float = 0.1,
        bias: float = 0.0,
    ) -> None:
        super().__init__()
        if min_steps < 0:
            raise ValueError("min_steps must be non-negative.")
        if max_steps < min_steps:
            raise ValueError("max_steps must be >= min_steps.")
        if not 0.0 <= continue_threshold <= 1.0:
            raise ValueError("continue_threshold must be in [0, 1].")

        self.min_steps = min_steps
        self.max_steps = max_steps
        self.continue_threshold = continue_threshold
        self._seed = int(seed)
        self._init_scale = float(init_scale)
        self._bias = float(bias)
        self._linear: nn.Linear | None = None
        self.last_probability: float = 1.0

    def _ensure_linear(self, dim: int, device: torch.device) -> None:
        if self._linear is not None and self._linear.in_features == dim:
            return

        linear = nn.Linear(dim, 1, bias=True, device=device)
        gen = torch.Generator()
        gen.manual_seed(self._seed)
        with torch.no_grad():
            weight = torch.randn(linear.weight.shape, generator=gen, dtype=linear.weight.dtype)
            linear.weight.copy_(weight.to(device=device) * self._init_scale)
            linear.bias.fill_(self._bias)
        self._linear = linear

    def continuation_probability(self, state: Any) -> float:
        state_vec = _state_to_vector(state)
        self._ensure_linear(state_vec.shape[0], state_vec.device)
        assert self._linear is not None
        logit = self._linear(state_vec)
        return float(torch.sigmoid(logit).item())

    def should_continue(self, state: Any, step_idx: int) -> bool:
        if step_idx < self.min_steps:
            self.last_probability = 1.0
            return True
        if step_idx >= self.max_steps:
            self.last_probability = 0.0
            return False

        probability = self.continuation_probability(state)
        self.last_probability = probability
        return probability >= self.continue_threshold


class RefinementHaltingPolicy(_SigmoidHaltingPolicy):
    """Decides whether to continue inner refinement."""

    def __init__(
        self,
        min_steps: int = 1,
        max_steps: int = 8,
        continue_threshold: float = 0.5,
        seed: int = 0,
        init_scale: float = 0.1,
        bias: float = 0.0,
    ) -> None:
        super().__init__(
            min_steps=min_steps,
            max_steps=max_steps,
            continue_threshold=continue_threshold,
            seed=seed,
            init_scale=init_scale,
            bias=bias,
        )


class SegmentHaltingPolicy(_SigmoidHaltingPolicy):
    """Decides whether to continue outer staged reasoning."""

    def __init__(
        self,
        min_steps: int = 1,
        max_steps: int = 6,
        continue_threshold: float = 0.5,
        seed: int = 1,
        init_scale: float = 0.1,
        bias: float = 0.0,
    ) -> None:
        super().__init__(
            min_steps=min_steps,
            max_steps=max_steps,
            continue_threshold=continue_threshold,
            seed=seed,
            init_scale=init_scale,
            bias=bias,
        )
