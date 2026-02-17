"""Pre-merge validator for the HRM Sudoku dataset loader.

Usage:
  # Validate a real HRM dataset folder
  python scripts/validate_sudoku_dataset.py --dataset-path data/sudoku-extreme-full

  # Run an isolated smoke test with a synthetic HRM-format dataset
  python scripts/validate_sudoku_dataset.py --self-test
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from looped_hrm.data.sudoku_dataset import SudokuDataset


def _write_subset(base_dir: Path, split: str, puzzles: np.ndarray, labels: np.ndarray) -> None:
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
    np.save(split_dir / "all__labels.npy", labels)


def _build_synthetic_dataset(dataset_root: Path) -> None:
    # HRM token space is [1..10], where 1 means blank and 2..10 map to digits 1..9.
    train_inputs = np.ones((2, 81), dtype=np.int32)
    train_labels = np.full((2, 81), 10, dtype=np.int32)

    # Easy sample (40 givens) + hard sample (20 givens).
    train_inputs[0, :40] = 10
    train_inputs[1, :20] = 10

    test_inputs = np.ones((1, 81), dtype=np.int32)
    test_inputs[0, :32] = 10  # medium
    test_labels = np.full((1, 81), 10, dtype=np.int32)

    _write_subset(dataset_root, "train", train_inputs, train_labels)
    _write_subset(dataset_root, "test", test_inputs, test_labels)


def _validate_dataset(dataset_path: Path) -> None:
    for split in ("train", "test", "val"):
        dataset = SudokuDataset(split=split, dataset_path=dataset_path)
        if len(dataset) == 0:
            raise ValueError(f"{split} split loaded but had zero samples.")

        sample = dataset[0]
        if len(sample.puzzle) != 9 or len(sample.puzzle[0]) != 9:
            raise ValueError(f"{split} sample puzzle is not 9x9.")
        if len(sample.solution) != 9 or len(sample.solution[0]) != 9:
            raise ValueError(f"{split} sample solution is not 9x9.")

        print(f"[{split}] examples={len(dataset)} first_difficulty={sample.difficulty}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate HRM Sudoku dataset compatibility")
    parser.add_argument("--dataset-path", type=Path, help="Path to HRM-style dataset root")
    parser.add_argument("--self-test", action="store_true", help="Run synthetic end-to-end smoke test")
    args = parser.parse_args()

    if args.self_test:
        with tempfile.TemporaryDirectory() as tmp:
            dataset_root = Path(tmp) / "synthetic-hrm-sudoku"
            _build_synthetic_dataset(dataset_root)
            _validate_dataset(dataset_root)
            print("Self-test passed.")
        return

    if args.dataset_path is None:
        raise ValueError("Provide --dataset-path for real data validation, or run with --self-test.")

    _validate_dataset(args.dataset_path)
    print("Dataset validation passed.")


if __name__ == "__main__":
    main()
