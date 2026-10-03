# Lab 4 — Monitoring and Production Deployment

**DDM501 — AI in DevOps, DataOps, MLOps · Session 9**

Lab 3 ended with a test suite that catches broken models before they ship.
This lab starts from the failure that suite cannot catch: a model that is
still correct, still passing every test, and quietly becoming wrong because
the world moved.

This repository is our team's completed version: the credit-default scoring
API instrumented for Prometheus, PSI drift and fairness monitoring over a live
window, seven unit-tested ML alert rules, and a provisioned Grafana dashboard.
The written analysis of the traffic profiles is in
[`docs/ANALYSIS.md`](docs/ANALYSIS.md) (PDF: `docs/ANALYSIS.pdf`).

---

## The problem

You cannot monitor accuracy in production.

Whether an applicant defaults is known next month at the earliest. For an
applicant you declined, it is never known at all — you refused them credit, so
there is no outcome to observe. By the time an accuracy number arrives, the
model has been deciding on the wrong distribution for weeks.

So production ML monitoring means watching **proxies that move before the
accuracy does**:

| Signal | What it answers | Metric |
|---|---|---|
| Input drift | Do the applicants still look like the training population? | `ml_feature_drift_psi` |
| Output drift | Has the distribution of scores shifted? | `ml_prediction_score` |
| Decision mix | Is more work being pushed to manual review? | `ml_decisions_total` |
| Fairness gap | Is one group being flagged at a different rate? | `ml_fairness_gap` |

None of these proves the model is wrong. Each is a reason to look.

---

## The thirteen tasks

| # | File | What | Owner |
|---|---|---|---|
| 1 | `app/monitoring.py` | `population_stability_index` | maxnguyen83 |
| 2 | `app/monitoring.py` | `MonitoringWindow.compute_drift` | maxnguyen83 |
| 3 | `app/monitoring.py` | `MonitoringWindow.compute_fairness` | maxnguyen83 |
| 4 | `app/monitoring.py` | `MonitoringWindow.publish` | maxnguyen83 |
| 5 | `app/main.py` | `_observe` — derive features before recording | Ducmanh2212 |
| 6 | `app/main.py` | `GET /metrics` | Ducmanh2212 |
| 7 | `app/main.py` | `GET /monitoring` | Ducmanh2212 |
| 8 | `app/main.py` | `POST /explain` | Ducmanh2212 |
| 9 | `app/middleware.py` | `MetricsMiddleware.dispatch` | Ducmanh2212 |
| 10 | `app/explain.py` | `Explainer.explain` | Ducmanh2212 |
| 11 | `scripts/make_reference.py` | `build_reference` | hieunt-fsb-ai |
| 12 | `monitoring/prometheus/alerts/ml_alerts.yml` | seven alert rules | hieunt-fsb-ai |
| 13 | `monitoring/grafana/dashboards/model-behaviour.json` | the dashboard | thientd2609 |

CI, the extra promtool tests and the Makefile are hieunt-fsb-ai's; the load
runs, screenshots, this README and the analysis are thientd2609's.

---

## Running it

The scripts import the `app` and `pipeline` packages from the repo root, so
the root has to be on `PYTHONPATH` (`python scripts/x.py` only puts `scripts/`
there). The Makefile and CI export it; by hand:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD

python scripts/make_dataset.py      # 30,000 rows, UCI credit-default schema (deterministic, seed 501)
python scripts/train_model.py       # ROC AUC 0.7473 on the held-out 20%
python scripts/make_reference.py    # models/reference.json, committed with the model
```

Tests and alert rules:

```bash
make test               # pytest with coverage (fails under 85%)
make promtool           # needs promtool on PATH
make promtool-docker    # same checks through the prom/prometheus:v2.51.2 image
```

The stack (api, prometheus, grafana, node-exporter):

```bash
make up                 # docker compose up -d --build
```

| What | Where |
|---|---|
| API docs | http://localhost:8000/docs |
| Raw metrics | http://localhost:8000/metrics |
| Monitoring as JSON | http://localhost:8000/monitoring |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 — `admin` / `admin` (dashboards in folder *DDM501*) |

Traffic profiles, 400 requests each. The drift window lives in the API's
memory, so `make reset` (restart the API) empties it between profiles:

```bash
make load        && make explain   # normal traffic, plus 10 /explain calls
make reset && make drift-mild      # --strength 0.05
make reset && make drift           # full shift
make reset && make unfair          # SEX=1 applications made riskier
make reset && make defaulted       # a data bug: every applicant arrives with AGE=35
make watch                         # /monitoring JSON every 2 s
```

If the API is published on another port, pass it through:
`make load API_URL=http://localhost:48000`.

---

## Results from our runs

400 requests per profile, window reset between runs. Score mean and decision
mix are the load generator's own count of the responses; PSI and the gap are
from `/monitoring` right after the run.

| Profile | Score mean | REVIEW | DECLINE | Drift score | Leading feature | Fairness gap |
|---|---|---|---|---|---|---|
| normal | 0.2355 | 14.2% | 8.0% | 0.0301 stable | AGE 0.030 | 0.019 |
| drifted 0.05 | 0.2437 | 14.0% | 8.5% | 0.1885 moderate | payment_ratio 0.189 | 0.015 |
| drifted 0.15 | 0.2609 | 16.2% | 9.8% | 1.1518 significant | payment_ratio 1.152 | 0.036 |
| drifted full | 0.5897 | 29.8% | 54.8% | 4.2510 significant | payment_ratio 4.251 | 0.038 |
| unfair | 0.3966 | 15.8% | 32.0% | 0.3526 significant | max_delay 0.353 | **0.722** |
| defaulted (data bug) | 0.2383 | 13.2% | 8.2% | 7.5900 significant | AGE 7.590 | 0.020 |

What the table shows, in one line each (the argument is in the analysis):

- At strength 0.05 the decision mix is indistinguishable from normal while PSI
  is already in the moderate band. Input monitoring moves first.
- Features react very differently to the same shift: `payment_ratio` leads
  every drifted run. `PAY_0` and `max_delay` stay at their normal values until
  the full shift, because the generator only adds arrears above strength 0.25.
- `unfair` and `drifted` both read "significant" on drift; only the selection
  rate per group (0.94 vs 0.22) says who absorbed the change.
- `defaulted` is a data bug, not drift: AGE alone at 7.59, every other feature
  and the decision mix at their normal values. Fix the pipeline, do not retrain.

Screenshots of the Model Behaviour dashboard under each profile are in
[`docs/screenshots/`](docs/screenshots/).

---

## Design decisions

**PSI floors both proportions at `EPSILON` (1e-4).** An empty bucket on either
side gives `log(0)`. The alternative, Laplace smoothing of the counts, also
works but changes every bucket a little and makes the "PSI of a distribution
against itself is exactly 0" property approximate.

**Quantile bins with open outer edges, built once from the training split.**
Equal-width bins on `LIMIT_BAL` would put 53% of our data in the first bucket
and 85% in the first two, hiding most shifts; quantile bins start roughly
equally populated (10 bins holding 6.6–13.6% each). Discrete features collapse to fewer bins after
`np.unique` (`PAY_0` and `max_delay` have 4). The reference is a property of the
model and is committed; rebuilding it from recent traffic would make slow
drift invisible.

**"Not measuring" never looks like "stable".** Below 200 rows the drift dict
is empty and `sufficient_data` is false. `publish()` also clears the
per-feature and per-group gauges before setting them, so a PSI from before a
restart, or a group that fell under 30 rows, disappears from `/metrics`
instead of being served as if it were current.

**Drift is recomputed on scrape, not per request,** and a failure in that
computation is logged rather than failing `/metrics`: a 500 on the scrape would
page `ServiceDown` for what is a monitoring bug, not a scoring outage.

**Middleware labels a 404 with no matching route as `unmatched`.** The route
template already bounds cardinality for real routes; without this, every URL a
scanner tries would become its own `endpoint` series.

**`/explain` reports the applicant's own values.** Derived ratios come from
`add_derived_features`; where a ratio is undefined (no bill, so no payment
ratio) we report the training median the imputer substituted, because that is
what the model used. One-hot columns report their 0/1 indicator.

**Alert rules** follow the four disciplines (`for:`, ratio, severity, a
description that says what to do). `PredictionErrorsRising` is the one
deliberate absolute rate: the ratio version already exists as `HighErrorRate`,
and a ratio against `ml_predictions_total` goes silent exactly when every call
fails (that labelled counter has no series until the first success). Every
rule carries a `signal` label (`leading`, `output`, `fairness`, `service`) so a
router can send ML alerts and service alerts to different people. Moderate and
significant drift both fire above 0.25; collapsing them is an Alertmanager
inhibit rule, which this stack does not include.

**Dashboard model panels use a fixed 2-minute rate window.** A 400-request run
takes about 15 seconds, so a 5-minute window blends consecutive profiles. Over
such a short burst `rate()` extrapolates each series separately, so the
dashboard's DECLINE share and mean score can differ from the exact numbers by a
few points; `/monitoring` and the load generator are the exact source.

---

## Changes outside the TODOs, and why

| Change | Why |
|---|---|
| `PYTHONPATH` exported in CI and the Makefile | `python scripts/train_model.py` failed with `ModuleNotFoundError: pipeline` on a clean checkout, so the CI test job could not have passed as shipped |
| `.dockerignore` | without it a local `.venv` (about 400 MB here) is sent as build context on every `docker compose build` |
| `monitoring/prometheus/tests/ml_alert_tests.yml` | `alert_tests.yml` covers four of our seven rules; this adds a firing and a non-firing case for `DecisionMixShift`, `PredictionErrorsRising` and `ExplanationLatencyHigh` |
| Makefile: `promtool-docker`, `reset`, `explain`, `defaulted`, `API_URL` | run promtool without installing it; reset the window between profiles; give the explanation panels data; point the load generator at a remapped port |
| `scripts/load_test.py --profile defaulted` | measures the "drift that is really a data bug" signature instead of only describing it |
| New tests: `test_monitoring_publish.py`, `test_psi_edge_cases.py`, `test_middleware.py`, `test_failure_paths.py`, `test_make_reference.py` | pin the behaviour above (stale gauges cleared, raising handler counted as 500, unmatched label, scrape survives a drift bug, reference builder edge cases) |

The provided tests and `alert_tests.yml` are unchanged.

Local Python was 3.12; the Docker image and CI use 3.11. All pins in
`requirements.txt` install unchanged on both.

---

## Verification

| Check | Result |
|---|---|
| `pytest` | 100 passed, coverage 94.78% of `app/` (gate 85%) |
| `promtool check rules` | api_alerts 5 rules, ml_alerts 7 rules, SUCCESS |
| `promtool test rules alert_tests.yml ml_alert_tests.yml` | SUCCESS, SUCCESS |
| `actionlint .github/workflows/ci.yml` | no findings |
| Stack | all three scrape targets `up`, 12 rules loaded and healthy, every Model Behaviour query returns data in all six profile runs |
| Same profiles against `uvicorn` without Docker | identical numbers to four decimals |

---

## Known limitations

- PSI is univariate: a change in the relationship between two features with
  both marginals unchanged scores zero.
- The window counts requests, not time. On a quiet service the last 400
  requests can be days old, and the PSI keeps describing them.
- The fairness signal is a selection-rate gap (demographic parity). It says
  nothing about whether the flagged applicants were riskier; that needs labels,
  which arrive a month late and never for declined applicants.
- The window and the reference live in one process. Several API replicas would
  each see a slice of the traffic; the drift would have to be computed from a
  shared store or from the score histogram in Prometheus.
