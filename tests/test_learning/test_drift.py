from trade.learning.drift import DriftDetector, DriftState


def test_no_drift_stable():
    d = DriftDetector().assess(0.01, 0.01, 0.002, 0.002, 0.05, sample_count=50)
    assert d.state == DriftState.NO_DRIFT


def test_critical_drift():
    d = DriftDetector().assess(0.05, -0.05, 0.002, 0.01, 0.6, sample_count=50)
    assert d.state == DriftState.CRITICAL_DRIFT


def test_insufficient_samples():
    d = DriftDetector().assess(0.05, -0.05, 0.002, 0.01, 0.6, sample_count=5)
    assert d.state == DriftState.NO_DRIFT
