"""Sudoku dataset interfaces and generation/loading placeholders."""

from dataclasses import dataclass
from typing import List


@dataclass
class SudokuExample:
    puzzle: List[List[int]]
    solution: List[List[int]]
    difficulty: str


class SudokuDataset:
    """Placeholder dataset abstraction for Sudoku experiments."""

    def __init__(self, split: str = "train") -> None:
        self.split = split

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> SudokuExample:
        raise NotImplementedError
