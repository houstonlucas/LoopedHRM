"""Hierarchical iterative solver with two-level adaptive compute."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch
from torch import nn

from looped_hrm.compute.halting import RefinementHaltingPolicy, SegmentHaltingPolicy


def _as_grid_tensor(puzzle: Any, device: torch.device) -> torch.Tensor:
    grid = torch.as_tensor(puzzle, dtype=torch.long, device=device)
    if grid.shape != (9, 9):
        raise ValueError(f"Expected puzzle shape (9, 9), got {grid.shape}.")
    if not torch.all((grid >= 0) & (grid <= 9)):
        raise ValueError("Puzzle values must be integers in [0, 9].")
    return grid


@dataclass
class SolverOutput:
    prediction: Any
    metrics: Dict[str, float]


class _Encoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(10, hidden_dim)
        self.position_embedding = nn.Embedding(81, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        position_ids = torch.arange(81, device=token_ids.device, dtype=torch.long)
        latent = self.token_embedding(token_ids) + self.position_embedding(position_ids)
        return self.layer_norm(latent)


class _RefinementCore(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.latent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def step(self, latent: torch.Tensor, upper_state: torch.Tensor) -> torch.Tensor:
        update = torch.tanh(self.latent_proj(latent) + self.context_proj(upper_state).unsqueeze(0))
        return self.layer_norm(latent + update)


class _UpperCore(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)
        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.to_latent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.upper_norm = nn.LayerNorm(hidden_dim)
        self.latent_norm = nn.LayerNorm(hidden_dim)

    def step(self, latent: torch.Tensor, upper_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = latent.mean(dim=0)
        upper_update = torch.tanh(self.pool_proj(pooled) + self.state_proj(upper_state))
        next_upper = self.upper_norm(upper_state + upper_update)

        latent_update = torch.tanh(self.to_latent_proj(next_upper)).unsqueeze(0)
        next_latent = self.latent_norm(latent + latent_update)
        return next_upper, next_latent


class _Decoder(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.output_proj = nn.Linear(hidden_dim, 9)

    def forward(self, latent: torch.Tensor, givens: torch.Tensor, given_mask: torch.Tensor) -> torch.Tensor:
        logits = self.output_proj(latent)
        pred_digits = logits.argmax(dim=1) + 1
        pred_digits[given_mask] = givens[given_mask]
        return pred_digits.reshape(9, 9)


class HierarchicalIterativeSolver(nn.Module):
    """Two-level staged reasoning model with adaptive halting."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

        model_cfg = config.get("model", {})
        halting_cfg = config.get("halting", {})

        hidden_dim = int(model_cfg.get("hidden_dim", config.get("hidden_dim", 64)))
        seed = int(config.get("seed", 0))
        torch.manual_seed(seed)
        self.device = torch.device(model_cfg.get("device", config.get("device", "cpu")))

        self.encoder = _Encoder(hidden_dim=hidden_dim)
        self.refinement_core = _RefinementCore(hidden_dim=hidden_dim)
        self.upper_core = _UpperCore(hidden_dim=hidden_dim)
        self.decoder = _Decoder(hidden_dim=hidden_dim)

        self.refinement_halt = RefinementHaltingPolicy(
            min_steps=int(halting_cfg.get("refinement_min_steps", config.get("refinement_min_steps", 1))),
            max_steps=int(halting_cfg.get("refinement_max_steps", config.get("refinement_max_steps", 8))),
            continue_threshold=float(
                halting_cfg.get("refinement_continue_threshold", config.get("refinement_continue_threshold", 0.5))
            ),
            seed=seed + 13,
            bias=float(halting_cfg.get("refinement_bias", 0.0)),
        )
        self.segment_halt = SegmentHaltingPolicy(
            min_steps=int(halting_cfg.get("segment_min_steps", config.get("segment_min_steps", 1))),
            max_steps=int(halting_cfg.get("segment_max_steps", config.get("segment_max_steps", 6))),
            continue_threshold=float(
                halting_cfg.get("segment_continue_threshold", config.get("segment_continue_threshold", 0.5))
            ),
            seed=seed + 17,
            bias=float(halting_cfg.get("segment_bias", 0.0)),
        )
        self.to(self.device)

    def forward(self, puzzle: Any) -> SolverOutput:
        """Run hierarchical latent reasoning and return prediction and metrics."""
        grid = _as_grid_tensor(puzzle, device=self.device)
        flat_tokens = grid.reshape(-1)
        given_mask = flat_tokens > 0

        latent = self.encoder(flat_tokens)
        upper_state = latent.mean(dim=0)

        total_refine_steps = 0
        segments = 0
        refinement_probs: list[float] = []
        segment_probs: list[float] = []

        while self.segment_halt.should_continue(upper_state, segments):
            segment_probs.append(self.segment_halt.last_probability)
            segments += 1

            upper_state, latent = self.upper_core.step(latent=latent, upper_state=upper_state)

            refine_steps_this_segment = 0
            while self.refinement_halt.should_continue(latent, refine_steps_this_segment):
                refinement_probs.append(self.refinement_halt.last_probability)
                latent = self.refinement_core.step(latent=latent, upper_state=upper_state)
                refine_steps_this_segment += 1
                total_refine_steps += 1

        prediction = self.decoder(latent=latent, givens=flat_tokens, given_mask=given_mask)
        metrics = {
            "refinement_steps": float(total_refine_steps),
            "segments": float(segments),
            "avg_refinement_per_segment": float(total_refine_steps / segments) if segments > 0 else 0.0,
            "mean_refinement_continue_prob": (
                float(sum(refinement_probs) / len(refinement_probs)) if refinement_probs else 0.0
            ),
            "mean_segment_continue_prob": float(sum(segment_probs) / len(segment_probs)) if segment_probs else 0.0,
        }
        return SolverOutput(prediction=prediction, metrics=metrics)
