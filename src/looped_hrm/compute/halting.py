"""Halting policy skeletons for refinement and segment control."""

from typing import Any


class RefinementHaltingPolicy:
    """Decides whether to continue inner refinement."""

    def should_continue(self, state: Any, step_idx: int) -> bool:
        raise NotImplementedError


class SegmentHaltingPolicy:
    """Decides whether to continue outer staged reasoning."""

    def should_continue(self, state: Any, segment_idx: int) -> bool:
        raise NotImplementedError
