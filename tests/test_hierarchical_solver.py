import torch

from looped_hrm.compute.halting import RefinementHaltingPolicy, SegmentHaltingPolicy
from looped_hrm.models.hierarchical_solver import HierarchicalIterativeSolver


def test_hierarchical_solver_forward_returns_valid_grid_and_metrics() -> None:
    puzzle = torch.zeros((9, 9), dtype=torch.long)
    puzzle[0, 0] = 5
    puzzle[4, 4] = 7

    solver = HierarchicalIterativeSolver(
        config={
            "seed": 123,
            "model": {"hidden_dim": 32},
            "halting": {
                "segment_min_steps": 1,
                "segment_max_steps": 3,
                "refinement_min_steps": 1,
                "refinement_max_steps": 4,
            },
        }
    )
    out = solver.forward(puzzle)

    assert tuple(out.prediction.shape) == (9, 9)
    assert torch.all((out.prediction >= 1) & (out.prediction <= 9))
    assert int(out.prediction[0, 0]) == 5
    assert int(out.prediction[4, 4]) == 7

    assert set(out.metrics.keys()) == {
        "refinement_steps",
        "segments",
        "avg_refinement_per_segment",
        "mean_refinement_continue_prob",
        "mean_segment_continue_prob",
    }
    assert 1 <= int(out.metrics["segments"]) <= 3
    assert out.metrics["refinement_steps"] >= out.metrics["segments"]


def test_refinement_halting_respects_min_and_max_steps() -> None:
    policy = RefinementHaltingPolicy(
        min_steps=2,
        max_steps=3,
        continue_threshold=1.0,
        seed=0,
    )
    state = torch.zeros((81, 8), dtype=torch.float32)

    assert policy.should_continue(state, 0) is True
    assert policy.should_continue(state, 1) is True
    assert policy.should_continue(state, 2) is False
    assert policy.should_continue(state, 3) is False


def test_segment_halting_respects_max_steps_when_always_continuing() -> None:
    policy = SegmentHaltingPolicy(
        min_steps=0,
        max_steps=2,
        continue_threshold=0.0,
        seed=0,
    )
    state = torch.zeros((16,), dtype=torch.float32)

    assert policy.should_continue(state, 0) is True
    assert policy.should_continue(state, 1) is True
    assert policy.should_continue(state, 2) is False
