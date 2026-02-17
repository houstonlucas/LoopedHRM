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


def test_hrm_token_decoding_matches_expected_values(tmp_path) -> None:
    # In HRM format: 1 -> blank(0 after decode), 10 -> digit 9 after decode.
    puzzle = np.ones((1, 81), dtype=np.int32)
    puzzle[0, 0] = 10
    puzzle[0, 1] = 2  # should decode to 1

    solution = np.full((1, 81), 10, dtype=np.int32)
    solution[0, 0] = 2

    _write_hrm_subset(tmp_path, "train", puzzle, solution)

    item = SudokuDataset(split="train", dataset_path=tmp_path)[0]
    assert item.puzzle[0][0] == 9
    assert item.puzzle[0][1] == 1
    assert item.puzzle[0][2] == 0
    assert item.solution[0][0] == 1
    assert item.solution[0][1] == 9


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


@pytest.mark.parametrize(
    "givens,expected",
    [
        (36, "easy"),
        (30, "medium"),
        (29, "hard"),
    ],
)
def test_difficulty_thresholds(givens: int, expected: str) -> None:
    puzzle = np.zeros((9, 9), dtype=np.int32)
    puzzle.flat[:givens] = 1
    assert SudokuDataset._infer_difficulty(puzzle) == expected


def test_missing_required_files_raise_file_not_found(tmp_path) -> None:
    (tmp_path / "train").mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError):
        SudokuDataset(split="train", dataset_path=tmp_path)


def test_invalid_metadata_sets_raises(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True, exist_ok=True)

    bad_metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 11,
        "seq_len": 81,
        "num_puzzle_identifiers": 1,
        "total_groups": 1,
        "mean_puzzle_examples": 1,
        "sets": ["not_all"],
    }
    (split_dir / "dataset.json").write_text(json.dumps(bad_metadata), encoding="utf-8")
    np.save(split_dir / "all__inputs.npy", np.ones((1, 81), dtype=np.int32))
    np.save(split_dir / "all__labels.npy", np.ones((1, 81), dtype=np.int32))

    with pytest.raises(ValueError, match="Unexpected HRM metadata format"):
        SudokuDataset(split="train", dataset_path=tmp_path)


def test_mismatched_input_label_shapes_raise(tmp_path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "pad_id": 0,
        "ignore_label_id": 0,
        "blank_identifier_id": 0,
        "vocab_size": 11,
        "seq_len": 81,
        "num_puzzle_identifiers": 1,
        "total_groups": 1,
        "mean_puzzle_examples": 1,
        "sets": ["all"],
    }
    (split_dir / "dataset.json").write_text(json.dumps(metadata), encoding="utf-8")
    np.save(split_dir / "all__inputs.npy", np.ones((2, 81), dtype=np.int32))
    np.save(split_dir / "all__labels.npy", np.ones((1, 81), dtype=np.int32))

    with pytest.raises(ValueError, match="mismatched shapes"):
        SudokuDataset(split="train", dataset_path=tmp_path)


def test_solution_with_blank_value_raises(tmp_path) -> None:
    # Labels decode via minus-one; an HRM value of 1 decodes to 0 and is invalid for solutions.
    puzzle = np.full((1, 81), 10, dtype=np.int32)
    solution = np.full((1, 81), 10, dtype=np.int32)
    solution[0, 5] = 1
    _write_hrm_subset(tmp_path, "train", puzzle, solution)

    with pytest.raises(ValueError, match="Grid values out of range"):
        SudokuDataset(split="train", dataset_path=tmp_path)
