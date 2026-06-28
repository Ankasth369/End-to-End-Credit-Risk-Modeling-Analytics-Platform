# Credit Risk Modeling — PD × EAD × LGD → Expected Loss

An end-to-end credit risk pipeline on the Lending Club consumer loan dataset: three
component models (Probability of Default, Exposure at Default, Loss Given Default)
combined into a portfolio Expected Loss estimate, served through an interactive
Streamlit app and a FastAPI scoring service, with a full model-validation suite
(calibration, population stability, SHAP explainability, baseline comparison,
monotonicity, stress testing, and a fair-lending sensitivity check).

## Table of Contents

- [About](#about)
- [Architecture](#architecture)
- [Key Results](#key-results)
- [Methodology](#methodology)
- [Model Validation & Governance](#model-validation--governance)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Kubernetes](#kubernetes)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Known Limitations](#known-limitations)

## About

Credit risk on a loan portfolio is conventionally decomposed as:

```
Expected Loss = PD × EAD × LGD × Loan Amount
```

- **PD** (Probability of Default) — will this borrower default?
- **EAD** (Exposure at Default) — how much of the loan balance is outstanding if they do?
- **LGD** (Loss Given Default) — of that exposure, how much is actually lost (not recovered)?

This project builds each component from the [Lending Club accepted loans dataset](https://www.kaggle.com/datasets) (2007–2018Q4),
validates them against real out-of-time data, and combines them into a portfolio-level
Expected Loss figure — then spends as much effort validating and stress-testing that
figure as it did building the models in the first place, because a model that looks
good on a single headline metric and nothing else is not a model you should trust.

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │         notebooks/01-07 (Kaggle)         │
                    │   EDA → preprocessing → feature eng. →   │
                    │      PD / EAD / LGD → Expected Loss      │
                    └────────────────────┬──────────────────────┘
                                         │ produces
                                         ▼
                    ┌─────────────────────────────────────────┐
                    │      models/*.joblib, *.json, *.csv      │
                    │   data/expected_loss_test_2016_2017.csv  │
                    └────────────────────┬──────────────────────┘
                                         │ loaded by
                       ┌─────────────────┴─────────────────┐
                       ▼                                   ▼
              ┌─────────────────┐                 ┌──────────────────┐
              │  credit_core.py │◄────shared───────│   credit_core.py │
              │  (preprocess /  │      logic       │                  │
              │   score / cal.) │                  │                  │
              └────────┬────────┘                  └─────────┬────────┘
                       ▼                                     ▼
              ┌─────────────────┐                 ┌──────────────────┐
              │     app.py      │                 │      api.py      │
              │ Streamlit UI    │                 │  FastAPI service │
              │ (8 pages)       │                 │  (REST scoring)  │
              └─────────────────┘                 └──────────────────┘
                       │                                     │
                       └──────────────┬──────────────────────┘
                                      ▼
                       docker-compose.yml (one image,
                          two services + healthchecks)
```

`credit_core.py` is the single source of truth for preprocessing, scoring, and
calibration — both the UI and the API import from it, so they can never silently
diverge. `monitoring.py` and `model_registry.py` run independently against the same
artifacts for drift alerting and versioned model tracking (MLflow).

## Key Results

| Component | Model | Key Metric |
|---|---|---|
| **PD** | XGBoost, 1,967 trees, 58 features | Test AUC **0.7251**, Gini 0.4502, KS 0.3257 |
| **EAD** | Term-based mean lookup (not a regression model) | 36mo → 58.05%, 60mo → 73.48% |
| **LGD** | LightGBM, 180 trees, 90 features (Optuna-tuned) | MAE **0.0641**, R² 0.0095, WAPE 0.0692 |

**Portfolio validation** (462,426 loans, 2016–2017 test vintage, $6.66B exposure):

| Metric | Raw (as originally validated) | After recalibration |
|---|---|---|
| Total predicted EL | $1.86B (27.86% rate) | ~$1.00B (13.57% rate) |
| Total realized loss | $1.18B (17.77% rate) | — |
| Predicted vs. realized | **156.7%** (overshoot) | **~85%** |
| Calibration ECE (out-of-time) | 0.2135 | **0.0061** |

The raw PD score systematically overstated risk. A 2-parameter Platt-scaling
correction — fit on 2016 loans, validated on **held-out 2017 loans it never saw** —
cut calibration error by 97% without retraining anything. This is wired into live
scoring in the app/API; the originally-validated raw numbers are kept in the saved
artifacts for traceability.

**Does the model beat simple heuristics?**

| Method | Features | AUC |
|---|---|---|
| FICO score alone | 1 | 0.5974 |
| LendingClub's own sub-grade | 1 | 0.6874 |
| Simple logistic regression | 4 | 0.6909 |
| **This project's XGBoost model** | 58 | **0.7251** |

Yes — and by enough to matter: an estimated ~13–15% more portfolio profit than any
of the simpler alternatives under a shared cost assumption (see [Model Information → Baseline Comparison](app.py)).

**Decision-threshold backtest** (462,426 loans, 2016–2017 holdout, actual outcomes;
see `models/threshold_backtest.json`):

| | Approve-all baseline | Model-gated @ threshold 0.60 |
|---|---|---|
| Net profit | $100.6M | $279.3M |
| Default rate (approved loans) | 23.23% | 16.44% |
| Approval rate | 100% | 74.5% |

Rejecting the riskiest 25.5% of applicants (by PD score) at the profit-optimal
threshold yields an estimated **+177.6% net profit** and a **6.8-point default-rate
reduction**, vs. approving everyone. This is a backtest simulation, not a live
production result — "revenue" is a simplified proxy (undiscounted installment
interest, sub_grade-imputed interest rate, no cost of capital or prepayment
modeling), and no A/B test or real deployment has been run.

## Methodology

**Data.** Lending Club accepted loans, 2007–2018Q4. 2018 vintage loans are excluded
(right-censored — most haven't resolved to a final outcome yet).

**Split.** Vintage-based, not random: **train = 2007–2015, test = 2016–2017**. This is
deliberately harder than a random split — it requires the model to generalize across
time, not just interpolate within a period it's already seen. It's also what makes
this project's numbers *lower* than several public benchmarks on this same dataset
that use random splits and inflate their scores as a result (see [Known Limitations](#known-limitations)
and the leakage note below).

**Leakage avoidance.** `grade`, `sub_grade` (as a raw label), and `int_rate` are
LendingClub's *own* internal risk assessment output — using them as model inputs
means the model isn't predicting risk, it's partially just reading LendingClub's
answer back. This project excludes them from the PD feature set. (A comparable public
repo on the same dataset was checked directly and does include `grade`/`int_rate` as
PD features — its reported AUC of 0.684 is still lower than this project's 0.7251,
despite the leakage.)

**PD** — XGBoost classifier, hyperparameter-tuned via Optuna. 58 engineered features
(financial ratios, credit bureau fields, smoothed target-encoded state default rate).

**EAD** — every regression approach tried (linear, random forest, XGBoost, LightGBM)
produced *negative* R². The root cause: origination-time features can't predict *when*
during the loan term a borrower will default, and default timing is what drives how
much balance remains outstanding. The fix is the industry-standard fallback: a
term-based mean lookup (36-month vs. 60-month loans have structurally different EAD).
This is disclosed as a real limitation, not hidden behind a plausible-looking model.

**LGD** — LightGBM regressor, Optuna-tuned (20 trials). Near-zero R² (0.0095) is
expected here too: recovery outcomes are driven by the collections/legal process
*after* default, which isn't observable at origination. The model still captures a
small amount of real signal (MAE 0.0641 on a [0,1] target).

**Expected Loss** — notebook 07 rebuilds the entire 2016–2017 test population *fresh*
from raw data (not reusing `test.csv`/`test_fe.csv`) so PD, EAD, and LGD all score the
exact same rows through consistent preprocessing, avoiding row-alignment bugs across
three models trained via different pipelines.

## Model Validation & Governance

Beyond the headline metrics, this project runs the checks a real model-risk-management
review would ask for:

| Check | Where | Finding |
|---|---|---|
| **Calibration** | Model Information → PD Model | ECE 0.2135 raw → 0.0061 after Platt scaling (validated out-of-time) |
| **Population Stability (PSI)** | Model Stability page | Score PSI 0.0081 (stable); `initial_list_status` PSI 0.3714 (real drift — matches a known LendingClub policy change) |
| **SHAP explainability** | Single Loan page, Model Information | Per-prediction waterfall charts; verified SHAP output is in log-odds space before labeling anything |
| **Baseline/challenger comparison** | Model Information → PD Model | Beats FICO-alone, sub-grade-alone, and a simple logistic regression on both AUC and estimated profit |
| **Lift / cumulative gains** | Model Information → PD Model | Top 10% riskiest loans capture 22.6% of actual defaults (2.26× lift) |
| **Monotonicity** | Model Stability page | 6/7 checked features behave as expected; `credit_history_months` does **not** — a genuine, disclosed finding, not smoothed over |
| **Stress testing** | Stress Testing page | Interactive DTI/income/utilization/inquiry shocks rescored live against a verified-representative 50,000-loan sample |
| **Fair lending sensitivity** | Model Stability page | Explicitly **not** a true disparate-impact test (no protected-class or BISG data exists in this dataset) — the feasible substitute measures how much PD depends on state-level signal alone |
| **Adverse action reasons** | Single Loan page | ECOA/Reg B-style plain-language reasons generated from SHAP; a real bug (today's scoring date leaking in as a "reason") was caught and fixed |
| **Model registry** | `model_registry.py` | PD/LGD models + validated metrics logged to MLflow (SQLite-backed, local) instead of being bare files with no version history |
| **Automated monitoring** | `monitoring.py` | Headless PSI/calibration check, writes timestamped reports and file-based alerts on threshold breach — runnable on a schedule, not just a dashboard a human has to open |

See [Known Limitations](#known-limitations) for what these checks *don't* cover.

## Project Structure

```
├── notebooks/
│   ├── 01_eda.ipynb                  EDA
│   ├── 02_preprocessing.ipynb        cleaning, leakage-column removal, vintage split
│   ├── 03_feature_engineering.ipynb  ratios, encodings, feature selection
│   ├── 04_pd_model.ipynb             XGBoost PD model, Optuna tuning
│   ├── 05_ead_model.ipynb            EAD — term-based lookup (regression failed)
│   ├── 06_lgd_model.ipynb            LightGBM LGD model, Optuna tuning
│   └── 07_expected_loss.ipynb        combines PD×EAD×LGD, portfolio validation
├── credit_core.py                    shared preprocessing/scoring/calibration logic
├── app.py                            Streamlit app (8 pages)
├── api.py                            FastAPI scoring service
├── monitoring.py                     headless PSI/calibration health checks
├── model_registry.py                 MLflow model registration
├── precompute_*.py                   one-time artifact builders (PSI, SHAP sample,
│                                      baseline comparison, fair-lending sample, config)
├── tests/                            pytest suite (50 tests)
├── models/                           trained model artifacts + metadata
├── data/                             datasets (large raw files gitignored)
├── Dockerfile, docker-compose.yml    containerized app + API
├── k8s/                              Kubernetes manifests (not live-verified — see Kubernetes section)
├── .github/workflows/ci.yml          test suite + API smoke test on push
└── requirements.txt / requirements-docker.txt
```

## Setup

**Requirements:** Python 3.11 (3.9+ should work), or Docker.

```bash
git clone <this-repo>
cd "Credit Risk Modeling"
pip install -r requirements.txt
```

The notebooks expect `data/df_model_raw.csv` (the cleaned base dataset produced by
notebook 01) and are designed to be run on Kaggle, then downloaded — the raw source
files are too large for this repo (`accepted_2007_to_2018Q4.csv` is 1.6GB). Trained
model artifacts in `models/` **are** committed (only ~24MB total) so the app, API, and
tests all work without rerunning any notebook.

## Usage

**Streamlit app:**
```bash
streamlit run app.py
```
Open `http://localhost:8501`. Eight pages: Dashboard, Validation Summary, Single Loan
Assessment, Batch Scoring, Portfolio Analytics, Model Stability (PSI), Stress Testing,
Model Information.

**FastAPI service:**
```bash
CREDIT_API_KEY=your-real-key uvicorn api:app --host 0.0.0.0 --port 8000
```
Interactive docs at `http://localhost:8000/docs`.

**Both, via Docker:**
```bash
docker compose up --build
```
Streamlit at `localhost:8501`, API at `localhost:8000`. Set a real key first:
```bash
export CREDIT_API_KEY=your-real-key   # picked up by docker-compose.yml
```
Without it, the API falls back to an insecure dev-only key and prints a loud warning
on startup — never rely on that outside local development.

**Monitoring / registry (one-off scripts, not services):**
```bash
python monitoring.py         # PSI + calibration check, writes a report and alerts
python model_registry.py     # registers current models into MLflow (SQLite-backed)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # view the registry
```

## Kubernetes

`k8s/` has manifests for both services: `Namespace`, `ConfigMap`, a `Secret`
*template* (never commit a real key in it — use the imperative `kubectl create
secret` command documented inside the file instead), `Deployment` + `Service`
for each of the API and Streamlit app, and an `HorizontalPodAutoscaler` for the
API.

**Honesty check:** these are written and structurally validated (YAML syntax,
consistent `kind`/`name`/`namespace` cross-references), but **not live-deployed
and verified** — this environment has no running cluster (no minikube/kind
installed, Docker Desktop's Kubernetes isn't enabled), and installing new
cluster tooling to test them was deliberately left as your call rather than
done automatically. Treat these as a reviewed starting point, not a
proven-working deployment, until you've actually applied them somewhere.

Kubernetes is genuinely more than this project needs today — Docker Compose
already covers "run these two services together." These manifests exist to
demonstrate the pattern (Deployments, Services, ConfigMap/Secret separation,
autoscaling) for when/if that changes, not because this workload requires
cluster-scale orchestration right now.

**To try it locally** (using [kind](https://kind.sigs.k8s.io/), the lightest
option since it runs entirely inside Docker, which this project already needs):

```bash
# 1. Build the image and load it into the cluster (kind can't see your local
#    Docker image cache directly -- it has to be loaded in explicitly)
docker build -t credit-risk:latest .
kind create cluster --name credit-risk
kind load docker-image credit-risk:latest --name credit-risk

# 2. Apply everything
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl create secret generic credit-risk-secrets --namespace credit-risk \
  --from-literal=CREDIT_API_KEY='some-real-random-key'
kubectl apply -f k8s/api-deployment.yaml -f k8s/api-service.yaml
kubectl apply -f k8s/streamlit-deployment.yaml -f k8s/streamlit-service.yaml
# HPA needs metrics-server; skip on kind unless you've installed it separately
kubectl apply -f k8s/api-hpa.yaml

# 3. Check status, then reach the services via port-forward
kubectl get pods -n credit-risk
kubectl port-forward -n credit-risk svc/credit-risk-api 8000:8000
kubectl port-forward -n credit-risk svc/credit-risk-streamlit 8501:8501

# 4. Tear down
kind delete cluster --name credit-risk
```

## API Reference

All endpoints except `/health` require an `X-API-Key` header.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check, no auth |
| `GET` | `/model/info` | Model metadata: AUC, feature counts, calibration status |
| `POST` | `/score/single` | Score one loan application (JSON body) |
| `POST` | `/score/batch` | Score a CSV of loans (multipart file upload, max 50,000 rows) |

```bash
curl -X POST http://localhost:8000/score/single \
  -H "X-API-Key: your-key" -H "Content-Type: application/json" \
  -d '{"loan_amnt":15000,"term":36,"sub_grade":"B3","annual_inc":65000,
       "fico_score":700,"dti":18.5}'
```

Returns raw and calibrated PD, EAD, LGD, Expected Loss, EL rate, and risk level.
Every request is audit-logged to `logs/api_audit.log` (JSON lines: timestamp, request
ID, key inputs, latency — no credential or PII beyond what was already in the request).

## Testing

```bash
pytest
```
50 tests: preprocessing correctness, calibration math (including that Platt scaling
preserves rank order), adverse-action reason filtering (with a regression test for a
real bug that was caught and fixed), model regression tests pinned against the actual
trained model file, and full API integration tests via `TestClient`.

CI (`.github/workflows/ci.yml`) runs the full suite plus an API smoke test on every
push/PR — configured and verified locally, but not yet connected to a remote, since
this repo hasn't been pushed anywhere.

## Known Limitations

- **Reject inference bias** — this dataset only contains loans LendingClub actually
  funded. The model has never seen a rejected applicant, so it ranks risk among
  already-approved loans; it hasn't been validated for a real point-of-application
  underwriting decision.
- **EAD carries no borrower-level signal** — it's a 2-value lookup by loan term, not a
  predictive model, because every regression attempt failed (see Methodology).
- **LGD has near-zero R²** — recovery outcomes are dominated by the collections
  process, which isn't observable at origination.
- **Single train/test split** — no k-fold or repeated out-of-time validation; every
  metric here reflects one specific 2007–2015 / 2016–2017 split.
- **Fair lending check is a proxy, not a real test** — genuine disparate-impact
  testing needs protected-class data or a BISG (Bayesian Improved Surname Geocoding)
  proxy, neither of which exists in this dataset. The geographic sensitivity check is
  the closest honest substitute, not a stand-in for the real thing.
- **No production hardening beyond what's here** — no k8s/orchestration, no
  authentication beyond a single static API key, no rate limiting, no request-level
  authorization/RBAC. This is a demonstration of the *engineering pattern*
  (shared core logic, tests, API, containerization, monitoring, registry), not a
  hardened production deployment.
