"""
PSI and drift edge cases on inputs the provided tests do not use.

A drift monitor fed NaN or None must report "no evidence", never a number
that looks like a shift.
"""

import numpy as np

from app.monitoring import (
    MONITORED_FEATURES,
    MonitoringWindow,
    ReferenceDistribution,
    population_stability_index,
)

EDGES = [-np.inf, 0.25, 0.5, 0.75, np.inf]
UNIFORM = [0.25, 0.25, 0.25, 0.25]


def _reference():
    return ReferenceDistribution(
        bins={f: list(EDGES) for f in MONITORED_FEATURES},
        expected={f: list(UNIFORM) for f in MONITORED_FEATURES},
    )


class TestPSIEdgeCases:
    def test_all_nan_input_is_zero_not_an_error(self):
        values = np.array([np.nan] * 20)
        assert population_stability_index(values, EDGES, UNIFORM) == 0.0

    def test_a_column_that_arrived_as_none_is_not_measured_as_zero(self):
        """None must be dropped, not coerced to 0.0 — a 0.0 would land in the
        first bucket and look like a real shift."""
        window = MonitoringWindow(reference=_reference(), min_size=10)
        for _ in range(40):
            row = {f: 0.4 for f in MONITORED_FEATURES}
            row["payment_ratio"] = None
            window.record(row, 0.2, 1)
        drift = window.compute_drift()
        assert drift["payment_ratio"] == 0.0               # nothing to compare
        assert drift["LIMIT_BAL"] > 0.25                   # all mass in one bin
