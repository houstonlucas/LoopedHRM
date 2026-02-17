import json

import numpy as np
import pytest

from looped_hrm.data.sudoku_dataset import SudokuDataset


def _write_hrm_subset(base_dir, split: str, puzzles: np.ndarray, solutions: np.ndarray) -> None:
    split_dir = base_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 11,
        "seq_len": 81,
        "num_puzzle_identifiers": 1,
        "total_groups": len(puzzles),
        "mean_puzzle_examples": 1,
        "sets": ["all"],
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")

    np.save(split_dir / "all__inputs.npy", puzzles)
    np.save(split_dir / "all__labels.npy", solutions)


def test_hrm_dataset_loads_and_decodes(tmp_path) -> None:
    # HRM format stores [0..9] digits as [1..10]
    puzzle = np.ones((1, 81), dtype=np.int32)
    puzzle[0, :40] = 10  # 40 givens -> easy

    solution = np.full((1, 81), 10, dtype=np.int32)
    _write_hrm_subset(tmp_path, "train", puzzle, solution)

    dataset = SudokuDataset(split="train", dataset_path=tmp_path)
    assert len(dataset) == 1

    item = dataset[0]
    assert item.difficulty == "easy"
    assert len(item.puzzle) == 9 and len(item.puzzle[0]) == 9
    assert len(item.solution) == 9 and len(item.solution[0]) == 9
    assert 0 <= min(min(r) for r in item.puzzle) <= 9
    assert 1 <= min(min(r) for r in item.solution) <= 9


def test_val_split_falls_back_to_test(tmp_path) -> None:
    puzzle = np.ones((1, 81), dtype=np.int32)
    puzzle[0, :32] = 10
    solution = np.full((1, 81), 10, dtype=np.int32)
    _write_hrm_subset(tmp_path, "test", puzzle, solution)

    dataset = SudokuDataset(split="val", dataset_path=tmp_path)
    assert len(dataset) == 1
    assert dataset[0].difficulty == "medium"


def test_invalid_split_raises(tmp_path) -> None:
    with pytest.raises(ValueError):
        SudokuDataset(split="dev", dataset_path=tmp_path)
