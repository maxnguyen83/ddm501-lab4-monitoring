"""
Tests for the HTTP middleware's failure paths.

The provided suite checks the happy path through the real app. These use a
throwaway app with the same middleware so a handler can be made to raise, and
they remove the series they create so the cardinality test in test_metrics.py
still sees only the real routes, whatever order the files run in.
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY
from app.middleware import UNMATCHED, MetricsMiddleware


def _value(name, **labels):
    for family in REGISTRY.collect():
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                return sample.value
    return 0.0


@pytest.fixture
def probe_app():
    app = FastAPI()
    app.add_middleware(MetricsMiddleware)

    @app.get("/boom")
    async def boom():
        raise RuntimeError("handler failed")

    @app.get("/items/{item_id}")
    async def item(item_id: int):
        return {"id": item_id}

    yield app

    for labels in (("GET", "/boom", "500"), ("GET", "/items/{item_id}", "200"),
                   ("GET", UNMATCHED, "404")):
        try:
            REQUEST_COUNT.remove(*labels)
            REQUEST_LATENCY.remove(*labels[:2])
        except KeyError:
            pass


def test_a_request_that_raises_is_still_counted_as_a_500(probe_app):
    """Recording only on success reports a healthy service during an outage."""
    client = TestClient(probe_app, raise_server_exceptions=False)
    before = _value("http_requests_total", endpoint="/boom", status="500")
    assert client.get("/boom").status_code == 500
    assert _value("http_requests_total", endpoint="/boom", status="500") == before + 1


def test_in_progress_gauge_is_balanced_after_failures(probe_app):
    client = TestClient(probe_app, raise_server_exceptions=False)
    before = _value("http_requests_in_progress")
    for _ in range(5):
        client.get("/boom")
        client.get("/items/1")
    assert _value("http_requests_in_progress") == before


def test_path_parameters_are_labelled_by_template(probe_app):
    client = TestClient(probe_app)
    for item_id in range(1, 6):
        client.get(f"/items/{item_id}")
    assert _value("http_requests_total", endpoint="/items/{item_id}", status="200") == 5
    assert _value("http_requests_total", endpoint="/items/3", status="200") == 0


def test_unknown_paths_share_one_label(probe_app):
    """A crawler walking random URLs must add one series, not one per URL."""
    client = TestClient(probe_app)
    before = _value("http_requests_total", endpoint=UNMATCHED, status="404")
    for path in ("/wp-admin", "/.env", "/random/123"):
        assert client.get(path).status_code == 404
    assert _value("http_requests_total", endpoint=UNMATCHED, status="404") == before + 3
    assert _value("http_requests_total", endpoint="/.env", status="404") == 0
