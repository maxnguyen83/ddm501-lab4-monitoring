"""
Failure paths of the service endpoints.

What the service does when a dependency is missing or broken is the part of
the monitoring that runs during an incident, so it is tested like the rest.
"""

import app.main as main


def test_explain_without_an_explainer_is_a_503(client, application, monkeypatch):
    monkeypatch.setattr(main, "explainer", None)
    assert client.post("/explain", json=application).status_code == 503


def test_a_failing_explanation_is_a_500_and_is_counted(client, application, monkeypatch):
    class Broken:
        def explain(self, frame):
            raise ValueError("shap failed")

    monkeypatch.setattr(main, "explainer", Broken())
    assert client.post("/explain", json=application).status_code == 500
    assert 'ml_prediction_errors_total{error_type="ValueError"' in client.get("/metrics").text


def test_a_broken_drift_computation_does_not_fail_the_scrape(client, monkeypatch):
    """A 500 on /metrics reads as ServiceDown. A bug in the drift code is not
    an outage of the scoring service and must not page as one."""

    def explode():
        raise RuntimeError("drift bug")

    monkeypatch.setattr(main.window, "publish", explode)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "ml_drift_score" in response.text


def test_a_failing_prediction_is_a_500_and_is_counted(client, application, monkeypatch):
    def broken(_payloads):
        raise KeyError("missing column")

    monkeypatch.setattr(main.model, "predict_proba", broken)
    assert client.post("/predict", json=application).status_code == 500
    assert client.post(
        "/predict/batch", json={"applications": [application]}
    ).status_code == 500
    assert 'ml_prediction_errors_total{error_type="KeyError"' in client.get("/metrics").text


def test_explanation_values_are_never_nan(client, application):
    """An applicant with no bill has no payment ratio. The model saw the
    imputed median; the explanation must report that number, not NaN (which
    JSON would serialise as null)."""
    import math

    no_bill = dict(application, bill_amt=[0.0] * 6, pay_amt=[0.0] * 6)
    frame = main.model.to_frame([no_bill])
    every = main.explainer.explain(frame, top_n=len(main.explainer.feature_names))
    values = {c["feature"]: c["value"] for c in every["contributions"]}
    assert all(math.isfinite(v) for v in values.values())
    assert 0.0 < values["payment_ratio"] <= 5.0       # the training median, not 0 or NaN

    body = client.post("/explain", json=no_bill).json()
    assert all(isinstance(c["value"], float) for c in body["contributions"])
