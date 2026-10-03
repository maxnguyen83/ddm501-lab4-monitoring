"""
Tests for what publish() pushes into Prometheus, beyond the shape of the JSON.

The gauges outlive any one window. These tests pin down that a value which is
no longer being measured disappears from the exposition instead of being
served forever as if it were current.
"""

import numpy as np
import pytest
from prometheus_client import REGISTRY

from app.monitoring import (
    MONITORED_FEATURES,
    MonitoringWindow,
    ReferenceDistribution,
)

EDGES = [-np.inf, 0.25, 0.5, 0.75, np.inf]
UNIFORM = [0.25, 0.25, 0.25, 0.25]


def _reference():
    return ReferenceDistribution(
        bins={f: list(EDGES) for f in MONITORED_FEATURES},
        expected={f: list(UNIFORM) for f in MONITORED_FEATURES},
    )


def _samples(name):
    return [
        s for family in REGISTRY.collect() for s in family.samples if s.name == name
    ]


def _fill(window, n, value, score, group):
    for _ in range(n):
        window.record({f: value for f in MONITORED_FEATURES}, score, group)


class TestPublishedGauges:
    def test_drift_gauges_match_the_json(self):
        window = MonitoringWindow(reference=_reference(), min_size=50)
        _fill(window, 100, 0.9, 0.2, 1)
        state = window.publish()
        published = {s.labels["feature"]: s.value for s in _samples("ml_feature_drift_psi")}
        assert set(published) == set(MONITORED_FEATURES)
        for feature, psi in state["feature_psi"].items():
            assert published[feature] == pytest.approx(psi, abs=1e-4)
        assert _samples("ml_drift_score")[0].value == pytest.approx(state["drift_score"], abs=1e-4)

    def test_per_feature_gauges_are_cleared_when_no_longer_measured(self):
        """After a restart the window is small again. A PSI left over from the
        previous window must not keep reporting as if it were live."""
        big = MonitoringWindow(reference=_reference(), min_size=50)
        _fill(big, 100, 0.9, 0.2, 1)
        big.publish()
        assert _samples("ml_feature_drift_psi")

        small = MonitoringWindow(reference=_reference(), min_size=50)
        _fill(small, 10, 0.9, 0.2, 1)
        state = small.publish()
        assert state["sufficient_data"] is False
        assert _samples("ml_feature_drift_psi") == []
        assert _samples("ml_drift_score")[0].value == 0.0

    def test_a_group_that_shrinks_below_the_minimum_disappears(self):
        window = MonitoringWindow(max_size=100, min_size=10, threshold=0.5)
        _fill(window, 50, 0.4, 0.9, 1)
        _fill(window, 50, 0.4, 0.1, 2)
        window.publish()
        assert {s.labels["group"] for s in _samples("ml_selection_rate")} == {"1", "2"}

        # Group 2 is pushed out of the bounded window by group 1 traffic.
        _fill(window, 80, 0.4, 0.9, 1)
        state = window.publish()
        assert set(state["selection_rate"]) == {"1"}
        assert {s.labels["group"] for s in _samples("ml_selection_rate")} == {"1"}
        assert state["fairness_gap"] == 0.0

    def test_gap_is_the_spread_across_all_groups(self):
        window = MonitoringWindow(min_size=10, threshold=0.5)
        _fill(window, 40, 0.4, 0.9, 1)                       # rate 1.0
        _fill(window, 20, 0.4, 0.9, 2)
        _fill(window, 20, 0.4, 0.1, 2)                       # rate 0.5
        _fill(window, 40, 0.4, 0.1, 3)                       # rate 0.0
        assert window.publish()["fairness_gap"] == pytest.approx(1.0)
