"""
FastAPI scoring service for the credit risk model.

Built on credit_core.py so this service and the Streamlit app share the
exact same preprocessing/scoring logic. Run with:

    uvicorn api:app --host 0.0.0.0 --port 8000

Auth: pass header `X-API-Key: <key>`. Set the CREDIT_API_KEY environment
variable in any real deployment -- the fallback below is dev-only and is
logged loudly on startup so it's never mistaken for a real secret.
"""
import io
import json
import os
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

import credit_core as cc

BASE = Path(__file__).parent
LOG_DIR = BASE / 'logs'
LOG_DIR.mkdir(exist_ok=True)
AUDIT_LOG_PATH = LOG_DIR / 'api_audit.log'

DEV_FALLBACK_KEY = "dev-only-insecure-key-change-me"
API_KEY = os.environ.get("CREDIT_API_KEY", DEV_FALLBACK_KEY)
if API_KEY == DEV_FALLBACK_KEY:
    print("=" * 70)
    print("WARNING: CREDIT_API_KEY is not set. Using an insecure dev-only key.")
    print("Set the CREDIT_API_KEY environment variable before deploying this.")
    print("=" * 70)

_state = {}


def load_artifacts():
    _state['models'] = cc.load_models(BASE)
    _state['config'] = cc.load_config(BASE)
    _state['platt'] = cc.load_calibration_model(BASE)
    _state['pd_explainer'] = None  # loaded lazily; shap import/build is not free
    print(f"Loaded models. PD AUC={_state['models']['pd_meta']['test_auc']:.4f}, "
          f"{len(_state['models']['pd_features'])} PD features.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield
    _state.clear()


app = FastAPI(
    title="Credit Risk Scoring API",
    description="PD x EAD x LGD Expected Loss scoring for Lending Club-style consumer loans.",
    version="1.0.0",
    lifespan=lifespan,
)


def get_pd_explainer():
    if _state.get('pd_explainer') is None:
        import shap
        _state['pd_explainer'] = shap.TreeExplainer(_state['models']['pd_model'])
    return _state['pd_explainer']


def check_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key is None or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header")
    return x_api_key


def audit_log(request_id, endpoint, n_loans, meta=None):
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'request_id': request_id,
        'endpoint': endpoint,
        'n_loans': n_loans,
        'meta': meta or {},
    }
    with open(AUDIT_LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + '\n')


# ── Schemas ──────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    loan_amnt: float = Field(..., gt=0, le=40000, description="Requested loan amount")
    term: int = Field(..., description="Loan term in months (36 or 60)")
    int_rate: float = Field(13.0, ge=5.0, le=31.0)
    sub_grade: str = Field("C3", description="Lending Club sub-grade, e.g. 'B3'")
    purpose: str = Field("debt_consolidation")
    annual_inc: float = Field(..., ge=0)
    emp_length: int = Field(5, ge=0, le=10)
    home_ownership: str = Field("RENT", description="RENT, MORTGAGE, OWN, or OTHER")
    addr_state: str = Field("CA", description="Two-letter US state code")
    verification_status: str = Field("Not Verified")
    credit_history_months: float = Field(120, ge=0)
    fico_score: float = Field(..., ge=300, le=850)
    dti: float = Field(..., ge=0)
    open_acc: float = Field(10, ge=0)
    total_acc: float = Field(20, ge=0)
    revol_bal: float = Field(10000, ge=0)
    revol_util: float = Field(50.0, ge=0)
    inq_last_6mths: float = Field(0, ge=0)
    delinq_2yrs: float = Field(0, ge=0)


class ScoreResponse(BaseModel):
    request_id: str
    pd_raw: float
    pd_calibrated: float
    ead: float
    lgd: float
    expected_loss: float
    el_rate: float
    risk_level: str
    model_pd_auc: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    pd_features: int
    pd_auc: Optional[float] = None


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    loaded = 'models' in _state
    return HealthResponse(
        status="ok" if loaded else "not_ready",
        models_loaded=loaded,
        pd_features=len(_state['models']['pd_features']) if loaded else 0,
        pd_auc=_state['models']['pd_meta']['test_auc'] if loaded else None,
    )


@app.get("/model/info")
def model_info(x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    m = _state['models']
    return {
        'pd_model': {'type': 'XGBoost', 'auc': m['pd_meta']['test_auc'],
                     'n_features': len(m['pd_features']), 'n_estimators': m['pd_meta']['n_estimators']},
        'ead_model': {'type': 'Term-based lookup', 'lookup': m['ead_lookup']},
        'lgd_model': {'type': 'LightGBM', 'mae': m['lgd_meta']['test_mae'],
                      'r2': m['lgd_meta']['test_r2'], 'n_features': len(m['lgd_features'])},
        'calibration': 'Platt scaling applied' if _state['platt'] is not None else 'none',
    }


@app.post("/score/single", response_model=ScoreResponse)
def score_single(loan: LoanApplication, x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    request_id = str(uuid.uuid4())
    t0 = time.time()

    inputs = loan.model_dump()
    inputs['installment'] = cc.compute_installment(inputs['loan_amnt'], inputs['term'], inputs['int_rate'])
    inputs['issue_year'] = datetime.now().year
    inputs['issue_month'] = datetime.now().month

    result = cc.score_single_loan(inputs, _state['models'], _state['config'], _state['platt'])
    risk_label, _ = cc.get_risk_level(result['el_rate'])

    audit_log(request_id, '/score/single', 1, {
        'loan_amnt': inputs['loan_amnt'], 'sub_grade': inputs['sub_grade'],
        'pd_calibrated': result['pd_calibrated'], 'expected_loss': result['expected_loss'],
        'latency_ms': round((time.time() - t0) * 1000, 1),
    })

    return ScoreResponse(
        request_id=request_id,
        pd_raw=result['pd_raw'], pd_calibrated=result['pd_calibrated'],
        ead=result['ead'], lgd=result['lgd'],
        expected_loss=result['expected_loss'], el_rate=result['el_rate'],
        risk_level=risk_label,
        model_pd_auc=_state['models']['pd_meta']['test_auc'],
    )


@app.post("/score/batch")
async def score_batch(file: UploadFile = File(...), x_api_key: str = Header(...)):
    check_api_key(x_api_key)
    request_id = str(uuid.uuid4())
    t0 = time.time()

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="File must be a .csv")

    contents = await file.read()
    try:
        df_raw = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    required = ['loan_amnt', 'term', 'sub_grade']
    missing = [c for c in required if c not in df_raw.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing}")

    if len(df_raw) > 50000:
        raise HTTPException(status_code=400, detail="Batch limited to 50,000 rows per request")

    models, config, platt = _state['models'], _state['config'], _state['platt']
    pd_X, lgd_X, ead_pct = cc.preprocess_batch(df_raw, config, models['pd_features'], models['lgd_features'])
    pd_pred, ead_pred, lgd_pred, _ = cc.score(pd_X, lgd_X, ead_pct, df_raw['loan_amnt'].values, models)
    pd_calibrated = cc.calibrate_pd(pd_pred, platt)
    el = pd_calibrated * ead_pred * lgd_pred * df_raw['loan_amnt'].values
    el_rate = el / df_raw['loan_amnt'].values

    results = df_raw.copy()
    results['pd_raw'] = pd_pred
    results['pd_calibrated'] = pd_calibrated
    results['ead'] = ead_pred
    results['lgd'] = lgd_pred
    results['expected_loss'] = el
    results['el_rate'] = el_rate
    results['risk_level'] = [cc.get_risk_level(r)[0] for r in el_rate]

    audit_log(request_id, '/score/batch', len(results), {
        'total_exposure': float(df_raw['loan_amnt'].sum()),
        'total_expected_loss': float(el.sum()),
        'latency_ms': round((time.time() - t0) * 1000, 1),
    })

    return {
        'request_id': request_id,
        'n_loans': len(results),
        'total_expected_loss': float(el.sum()),
        'avg_pd_calibrated': float(pd_calibrated.mean()),
        'results': json.loads(results.to_json(orient='records')),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
