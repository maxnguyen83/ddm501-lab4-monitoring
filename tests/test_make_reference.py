"""
Tests for build_reference on frames where the answer is known in advance.

tests/test_monitoring.py checks the reference file that was actually built.
These check the builder itself, including the cases the training data does
not happen to exercise: a constant column and a missing one.
"""

import math

import numpy as np
import pandas as pd
import pytest

from app.monitoring import MONITORED_FEATURES, population_stability_index
from scripts.make_reference import build_reference


@pytest.fixture
def frame():
    rng = np.random.default_rng(3)
    n = 5000
    return pd.DataFrame({
        "LIMIT_BAL": rng.lognormal(11, 0.8, n),             # skewed
        "AGE": rng.integers(21, 70, n),
        "PAY_0": rng.choice([-2, -1, 0, 1, 2], n, p=[0.1, 0.2, 0.5, 0.15, 0.05]),
        "utilisation_ratio": rng.random(n),
        "payment_ratio": np.where(rng.random(n) < 0.1, np.nan, rng.random(n)),
        "max_delay": np.zeros(n),                           # constant
    })


def test_outer_edges_are_open_and_proportions_sum_to_one(frame):
    reference = build_reference(frame, n_bins=10)
    for feature, edges in reference["bins"].items():
        assert edges[0] == -math.inf and edges[-1] == math.inf, feature
        assert sum(reference["expected"][feature]) == pytest.approx(1.0)
        assert len(reference["expected"][feature]) == len(edges) - 1


def test_quantile_bins_are_equally_populated_on_a_skewed_feature(frame):
    expected = build_reference(frame, n_bins=10)["expected"]["LIMIT_BAL"]
    assert max(expected) - min(expected) < 0.01


def test_a_constant_feature_is_skipped_rather_than_binned(frame):
    """Two buckets is noise, and a constant has fewer still."""
    assert "max_delay" not in build_reference(frame)["bins"]


def test_missing_features_are_skipped(frame):
    reference = build_reference(frame.drop(columns=["AGE"]))
    assert "AGE" not in reference["bins"]
    assert set(reference["bins"]) <= set(MONITORED_FEATURES)


def test_nan_is_dropped_not_binned_as_zero(frame):
    reference = build_reference(frame)
    values = frame["payment_ratio"].dropna().to_numpy()
    psi = population_stability_index(
        values, reference["bins"]["payment_ratio"], reference["expected"]["payment_ratio"]
    )
    assert psi < 0.001


def test_the_reference_survives_a_json_round_trip(frame):
    """-inf/+inf are not standard JSON. Python writes them as -Infinity and
    Infinity and reads them back; this pins that the file stays loadable."""
    import json

    reference = build_reference(frame)
    loaded = json.loads(json.dumps(reference))
    assert loaded["bins"]["AGE"][0] == -math.inf
