import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["LOG_PREDICTIONS"] = "0"

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from churnlib import config as C

EXAMPLE = {
    "credit_score": 619, "country": "Germany", "gender": "Female", "age": 52, "tenure": 4, "balance": 120000.0,
    "products_number": 3, "credit_card": 1, "active_member": 0, "estimated_salary": 85000.0,
    "txn_counts": [14, 13, 12, 8, 6, 4], "txn_amounts": [1200, 1100, 1000, 700, 520, 300],
    "login_count_30d": 2, "days_since_last_login": 21, "days_since_last_txn": 24, "complaints_6m": 2,
    "atm_share": 0.35, "mobile_share": 0.15, "branch_share": 0.50, "preferred_channel": "Branch"}
HEALTHY = {**EXAMPLE, "country": "France", "gender": "Male", "age": 30, "products_number": 2, "active_member": 1,
           "txn_counts": [20, 21, 20, 22, 21, 22], "txn_amounts": [1500] * 6, "login_count_30d": 25,
           "days_since_last_login": 0, "days_since_last_txn": 1, "complaints_6m": 0, "mobile_share": 0.7, "atm_share": 0.2, "branch_share": 0.1}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health"); assert r.status_code == 200 and r.json()["status"] == "ok"


def test_predict_risky_vs_healthy(client):
    risky = client.post("/predict", json=EXAMPLE).json()
    ok = client.post("/predict", json=HEALTHY).json()
    assert 0 <= risky["churn_probability"] <= 1
    assert risky["churn_probability"] > ok["churn_probability"] + 0.3
    assert risky["risk_band"] == "High" and ok["risk_band"] == "Low"
    assert len(risky["top_factors"]) == 5 and risky["recommended_action"]


def test_validation_errors(client):
    assert client.post("/predict", json={**EXAMPLE, "age": 12}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "txn_counts": [1, 2, 3]}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "mobile_share": 0.9}).status_code == 422   # shares sum != 1
    assert client.post("/predict", json={**EXAMPLE, "country": "Italy"}).status_code == 422


def test_missing_login_is_handled(client):
    r = client.post("/predict", json={**EXAMPLE, "login_count_30d": None, "days_since_last_login": None})
    assert r.status_code == 200


def test_customer_lookup(client):
    cid = int(pd.read_csv(C.TEST_FILE).customer_id.iloc[0])
    r = client.get(f"/customer/{cid}"); assert r.status_code == 200 and r.json()["customer_id"] == cid
    assert client.get("/customer/1").status_code == 404


def test_batch(client):
    r = client.post("/predict/batch", json=[EXAMPLE, HEALTHY]); assert r.status_code == 200 and len(r.json()) == 2


def test_service_matches_training_pipeline():
    """Serving must give the same probabilities as the training pipeline (no train/serve skew)."""
    import joblib
    from churnlib.service import ChurnService
    test = pd.read_csv(C.TEST_FILE).head(200)
    pipe = joblib.load(C.MODELS / "churn_pipeline.joblib")
    s = ChurnService(log_predictions=False)
    a = pipe.predict_proba(test)[:, 1]
    b = s.score_df(test.drop(columns=["login_missing_flag"]).assign(
        login_count_30d=test.login_count_30d.where(test.login_missing_flag == 0),
        days_since_last_login=test.days_since_last_login.where(test.login_missing_flag == 0))).churn_probability.to_numpy()
    assert abs(a - b).max() < 0.02          # tiny differences only from re-imputing the missing logins
