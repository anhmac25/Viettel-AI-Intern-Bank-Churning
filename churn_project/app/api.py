"""FastAPI churn service.

Run locally:   uvicorn app.api:app --reload
Interactive docs: http://localhost:8000/docs
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, model_validator

from churnlib import config as C
from churnlib.service import ChurnService

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["svc"] = ChurnService()
    yield
    state.clear()


app = FastAPI(
    title="Digital Bank - Customer Churn & Segmentation API",
    version=C.MODEL_VERSION,
    description="Scores a customer for churn risk, assigns a behavioural segment (persona), explains the main drivers and "
                "recommends a retention action. Built on public (Kaggle) + synthetic data - a demonstration, not a production bank system.",
    lifespan=lifespan,
)

Six = lambda t: Annotated[list[t], Field(min_length=6, max_length=6)]


class CustomerInput(BaseModel):
    credit_score: int = Field(ge=300, le=850)
    country: Literal["France", "Germany", "Spain"]
    gender: Literal["Female", "Male"]
    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0, le=10, description="years as a customer")
    balance: float = Field(ge=0, description="EUR")
    products_number: int = Field(ge=1, le=4)
    credit_card: Literal[0, 1]
    active_member: Literal[0, 1]
    estimated_salary: float = Field(ge=0, description="EUR per year")
    txn_counts: Six(int) = Field(description="transactions per month, oldest month first (6 values)")
    txn_amounts: Six(float) = Field(description="EUR per month, oldest month first (6 values)")
    login_count_30d: Optional[float] = Field(default=None, ge=0, description="null = no app tracking (imputed)")
    days_since_last_login: Optional[float] = Field(default=None, ge=0)
    days_since_last_txn: int = Field(ge=0, le=365)
    complaints_6m: int = Field(ge=0, le=50)
    atm_share: float = Field(ge=0, le=1)
    mobile_share: float = Field(ge=0, le=1)
    branch_share: float = Field(ge=0, le=1)
    preferred_channel: Optional[Literal["ATM", "Mobile app", "Branch"]] = None

    @model_validator(mode="after")
    def _checks(self):
        if any(c < 0 for c in self.txn_counts) or any(a < 0 for a in self.txn_amounts):
            raise ValueError("txn_counts and txn_amounts must be >= 0")
        if abs(self.atm_share + self.mobile_share + self.branch_share - 1) > 0.02:
            raise ValueError("atm_share + mobile_share + branch_share must sum to 1")
        if self.preferred_channel is None:
            shares = {"ATM": self.atm_share, "Mobile app": self.mobile_share, "Branch": self.branch_share}
            self.preferred_channel = max(shares, key=shares.get)
        return self

    model_config = {"json_schema_extra": {"example": {
        "credit_score": 619, "country": "Germany", "gender": "Female", "age": 52, "tenure": 4, "balance": 120000.0,
        "products_number": 3, "credit_card": 1, "active_member": 0, "estimated_salary": 85000.0,
        "txn_counts": [14, 13, 12, 8, 6, 4], "txn_amounts": [1200, 1100, 1000, 700, 520, 300],
        "login_count_30d": 2, "days_since_last_login": 21, "days_since_last_txn": 24, "complaints_6m": 2,
        "atm_share": 0.35, "mobile_share": 0.15, "branch_share": 0.50, "preferred_channel": "Branch"}}}


def to_record(c: CustomerInput) -> dict:
    d = c.model_dump()
    rec = {k: d[k] for k in C.PROFILE_COLS + ["login_count_30d", "days_since_last_login", "days_since_last_txn",
                                               "complaints_6m", "atm_share", "mobile_share", "branch_share", "preferred_channel"]}
    for i in range(6):
        rec[f"txn_count_m{i + 1}"] = d["txn_counts"][i]
        rec[f"txn_amount_m{i + 1}"] = d["txn_amounts"][i]
    return rec


def svc() -> ChurnService:
    if "svc" not in state:
        raise HTTPException(503, "model not loaded")
    return state["svc"]


@app.get("/", tags=["info"])
def root():
    return {"service": "churn-segmentation-api", "version": C.MODEL_VERSION, "docs": "/docs",
            "endpoints": ["/health", "/model-info", "POST /predict", "POST /predict/batch", "/customer/{customer_id}"]}


@app.get("/health", tags=["info"])
def health():
    s = svc()
    return {"status": "ok", "model_version": s.version, "algorithm": s.meta["algorithm"]}


@app.get("/model-info", tags=["info"])
def model_info():
    m = svc().meta
    return {k: m[k] for k in ["model_version", "trained_at", "algorithm", "threshold", "band_medium_from", "band_high_from",
                              "metrics_test_at_business_threshold", "metrics_test_at_0_5", "cv_roc_auc", "business"]}


@app.post("/predict", tags=["scoring"])
def predict(customer: CustomerInput, explain: bool = Query(True, description="include the top risk drivers")):
    return svc().score_record(to_record(customer), explain=explain)


@app.post("/predict/batch", tags=["scoring"])
def predict_batch(customers: Annotated[list[CustomerInput], Field(min_length=1, max_length=1000)]):
    df = pd.DataFrame([to_record(c) for c in customers])
    out = svc().score_df(df)
    return out.round(4).to_dict(orient="records")


@app.get("/customer/{customer_id}", tags=["scoring"])
def customer_lookup(customer_id: int, explain: bool = True):
    out = svc().lookup(customer_id, explain=explain)
    if out is None:
        raise HTTPException(404, f"customer {customer_id} not found")
    return out
