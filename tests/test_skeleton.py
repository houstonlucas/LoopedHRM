from looped_hrm.evaluation.metrics import compute_efficiency


def test_compute_efficiency_positive() -> None:
    assert compute_efficiency(0.9, 3.0) == 0.3
