"""
HTTP metrics middleware.

"""

import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.metrics import REQUEST_COUNT, REQUEST_LATENCY, REQUESTS_IN_PROGRESS

# Endpoint label for requests that matched no route. One fixed value, so a
# crawler walking random URLs adds one series, not one per URL.
UNMATCHED = "unmatched"


class MetricsMiddleware(BaseHTTPMiddleware):
    """Records count, latency and in-flight requests for every call."""

    # Scraping /metrics would otherwise count itself, and Prometheus polls it
    # every ten seconds — enough to dominate the request count on a quiet
    # service and make the traffic panel meaningless.
    EXCLUDED = {"/metrics"}

    async def dispatch(self, request: Request, call_next) -> Response:
        """TASK record count and latency for every request.

        Four things this must get right, each of which is a real outage
        someone has had:

          1. Skip the paths in EXCLUDED. Prometheus scrapes /metrics every
             ten seconds; counting those dominates the traffic panel on a
             quiet service.
          2. Label with the ROUTE TEMPLATE, not the resolved path — use the
             _route_template helper below. Without it, an endpoint like
             /applications/{id} mints one time series per id, and the
             cardinality explosion takes Prometheus down with it.
          3. Record in a `finally` block, so a request that raises on the way
             through is still counted. Middleware that only records on
             success reports a perfectly healthy service during an outage.
          4. Keep REQUESTS_IN_PROGRESS balanced: inc before, dec in the
             `finally`. An unbalanced gauge drifts upward forever.
        """
        if request.url.path in self.EXCLUDED:
            return await call_next(request)

        # Until a response exists, assume the worst: if call_next raises, the
        # client gets a 500, and that is what must be counted.
        status = "500"
        REQUESTS_IN_PROGRESS.inc()
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = str(response.status_code)
            return response
        finally:
            elapsed = time.perf_counter() - start
            # The route is only known after the router has run, so the
            # template is read here, not before call_next. A 404 matched no
            # route at all: every scanner probe would otherwise become its
            # own series, so they all share one label.
            fallback = UNMATCHED if status == "404" else request.url.path
            endpoint = _route_template(request, fallback)
            REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=status).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(elapsed)
            REQUESTS_IN_PROGRESS.dec()


def _route_template(request: Request, fallback: str) -> str:
    """The matched route's path template, or the raw path if none matched."""
    route = request.scope.get("route")
    return getattr(route, "path", fallback) or fallback
