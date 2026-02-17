from looped_hrm.evaluation.metrics import compute_efficiency, summarize_metrics


def test_compute_efficiency_divides_solved_rate_by_compute() -> None:
    assert compute_efficiency(0.9, 3.0) == 0.3


def test_compute_efficiency_non_positive_compute_returns_zero() -> None:
    assert compute_efficiency(0.9, 0.0) == 0.0
    assert compute_efficiency(0.9, -2.0) == 0.0


def test_summarize_metrics_passes_values_through() -> None:
    metrics = {"solved_rate": 0.8, "avg_compute": 5.0}
    assert summarize_metrics(metrics) == metrics