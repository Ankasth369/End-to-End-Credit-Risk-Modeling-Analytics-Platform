import io
import os

import pytest
from fastapi.testclient import TestClient

os.environ.setdefault("CREDIT_API_KEY", "test-key-for-pytest")

import api  # noqa: E402  (must import after setting env var)

HEADERS = {"X-API-Key": "test-key-for-pytest"}

SAMPLE_LOAN = {
    "loan_amnt": 15000, "term": 36, "int_rate": 12.5, "sub_grade": "B3",
    "purpose": "debt_consolidation", "annual_inc": 65000, "emp_length": 5,
    "home_ownership": "RENT", "addr_state": "CA", "verification_status": "Not Verified",
    "credit_history_months": 180, "fico_score": 700, "dti": 18.5,
    "open_acc": 11, "total_acc": 25, "revol_bal": 15000, "revol_util": 55.0,
    "inq_last_6mths": 1, "delinq_2yrs": 0,
}


@pytest.fixture(scope="module")
def client():
    with TestClient(api.app) as c:
        yield c


def test_health_no_auth_required(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] == 'ok'
    assert body['models_loaded'] is True
    assert body['pd_features'] == 58


def test_model_info_requires_auth(client):
    resp = client.get("/model/info")
    assert resp.status_code in (401, 422)  # missing header


def test_model_info_rejects_wrong_key(client):
    resp = client.get("/model/info", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


def test_model_info_with_valid_key(client):
    resp = client.get("/model/info", headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body['pd_model']['type'] == 'XGBoost'
    assert 0 < body['pd_model']['auc'] < 1


def test_score_single_requires_auth(client):
    resp = client.post("/score/single", json=SAMPLE_LOAN)
    assert resp.status_code in (401, 422)


def test_score_single_returns_sensible_output(client):
    resp = client.post("/score/single", json=SAMPLE_LOAN, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body['pd_raw'] <= 1
    assert 0 <= body['pd_calibrated'] <= 1
    assert body['ead'] in (0.5804732715289598, 0.7348155329774531)
    assert 0 <= body['lgd'] <= 1
    assert body['expected_loss'] > 0
    assert body['risk_level'] in ('Low', 'Medium', 'High', 'Very High')
    assert 'request_id' in body


def test_score_single_rejects_invalid_loan_amount(client):
    bad_loan = dict(SAMPLE_LOAN, loan_amnt=-100)
    resp = client.post("/score/single", json=bad_loan, headers=HEADERS)
    assert resp.status_code == 422


def test_score_single_worse_credit_scores_higher_pd(client):
    good = dict(SAMPLE_LOAN, fico_score=780, dti=8.0)
    bad = dict(SAMPLE_LOAN, fico_score=620, dti=35.0)
    resp_good = client.post("/score/single", json=good, headers=HEADERS).json()
    resp_bad = client.post("/score/single", json=bad, headers=HEADERS).json()
    assert resp_bad['pd_calibrated'] > resp_good['pd_calibrated']


def test_score_batch_requires_auth(client):
    csv_content = "loan_amnt,term,sub_grade\n15000,36,B3\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp = client.post("/score/batch", files=files)
    assert resp.status_code in (401, 422)


def test_score_batch_scores_multiple_loans(client):
    csv_content = (
        "loan_amnt,term,sub_grade,purpose,home_ownership,verification_status,"
        "addr_state,annual_inc,dti,fico_score\n"
        "15000,36,B3,debt_consolidation,RENT,Not Verified,CA,65000,18.5,700\n"
        "25000,60,D2,credit_card,MORTGAGE,Verified,NY,85000,22.0,680\n"
        "8000,36,A4,home_improvement,OWN,Source Verified,TX,45000,12.5,740\n"
    )
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp = client.post("/score/batch", files=files, headers=HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body['n_loans'] == 3
    assert len(body['results']) == 3
    assert body['total_expected_loss'] > 0


def test_score_batch_rejects_non_csv(client):
    files = {"file": ("test.txt", io.BytesIO(b"not a csv"), "text/plain")}
    resp = client.post("/score/batch", files=files, headers=HEADERS)
    assert resp.status_code == 400


def test_score_batch_rejects_missing_required_columns(client):
    csv_content = "loan_amnt,foo\n15000,bar\n"
    files = {"file": ("test.csv", io.BytesIO(csv_content.encode()), "text/csv")}
    resp = client.post("/score/batch", files=files, headers=HEADERS)
    assert resp.status_code == 400


def test_audit_log_written(client):
    log_lines_before = 0
    if api.AUDIT_LOG_PATH.exists():
        log_lines_before = len(api.AUDIT_LOG_PATH.read_text().splitlines())
    client.post("/score/single", json=SAMPLE_LOAN, headers=HEADERS)
    log_lines_after = len(api.AUDIT_LOG_PATH.read_text().splitlines())
    assert log_lines_after == log_lines_before + 1
