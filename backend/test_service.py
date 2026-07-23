"""Unit tests for the model and monitoring components."""

from model import IrisModel, CLASS_NAMES
from monitoring import Monitor


def test_model_trains_and_predicts():
    m = IrisModel()
    result = m.train()
    assert result.accuracy > 0.8  # Iris is easy; RF should score high
    out = m.predict([5.1, 3.5, 1.4, 0.2])  # a classic setosa sample
    assert out["prediction"] in CLASS_NAMES
    assert 0.0 <= out["confidence"] <= 1.0
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-6


def test_model_rejects_bad_input():
    m = IrisModel()
    m.train()
    try:
        m.predict([1.0, 2.0])  # wrong number of features
        assert False, "should have raised"
    except ValueError:
        pass


def test_monitor_tracks_metrics():
    mon = Monitor()
    mon.record(10.0, "setosa", error=False)
    mon.record(20.0, "versicolor", error=False)
    mon.record(0.0, None, error=True)
    snap = mon.snapshot()
    assert snap["total_requests"] == 3
    assert snap["errors"] == 1
    assert snap["prediction_distribution"]["setosa"] == 1
    assert snap["avg_latency_ms"] == 15.0
