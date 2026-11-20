"""Scoring service (Module 6). Loads the saved artefacts and scores customers.

Used by BOTH the FastAPI service and the Streamlit app, so every channel gives identical answers.
Includes prediction logging (input to the monitoring module) and a model-agnostic explanation
(two-point Shapley approximation over groups of inputs, see ChurnService.explain).
"""
import json
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .cleaning import apply_row_rules

GROUPS = {
    "Transaction activity (monthly counts)": C.TXN_COUNT_COLS,
    "Spending (monthly amounts)": C.TXN_AMOUNT_COLS,
    "Time since last transaction": ["days_since_last_txn"],
    "App usage (logins)": ["login_count_30d", "days_since_last_login", "login_missing_flag"],
    "Complaints": ["complaints_6m"],
    "Channel mix": ["atm_share", "mobile_share", "branch_share", "preferred_channel"],
    "Age": ["age"], "Country": ["country"], "Gender": ["gender"],
    "Number of products": ["products_number"], "Active-member status": ["active_member"],
    "Account balance": ["balance"], "Estimated salary": ["estimated_salary"],
    "Credit score": ["credit_score"], "Tenure": ["tenure"], "Credit card": ["credit_card"],
}


class ChurnService:
    def __init__(self, models_dir=C.MODELS, customers_file=C.MERGED_CLEAN, log_predictions=None):
        d = models_dir
        self.pipe = joblib.load(d / "churn_pipeline.joblib")
        self.meta = json.loads((d / "churn_metadata.json").read_text())
        self.params = json.loads((d / "cleaning_params.json").read_text())
        self.ref = json.loads((d / "reference_profile.json").read_text())
        seg = joblib.load(d / "segmenter.joblib")
        self.segmenter, self.cluster_to_persona = seg["segmenter"], seg["cluster_to_persona"]
        self.personas = json.loads((d / "personas.json").read_text())
        self.fb = self.pipe.named_steps["features"]
        self._customers_file = customers_file
        self._customers = None
        self.version = self.meta["model_version"]
        self.thr = float(self.meta["threshold"])
        self.t_medium = float(self.meta["band_medium_from"])
        self.t_high = float(self.meta["band_high_from"])
        if log_predictions is None:
            log_predictions = os.environ.get("LOG_PREDICTIONS", "1") == "1"
        self.log_predictions = log_predictions
        self.log_path = C.LOGS / "predictions.jsonl"

    # ------------------------------------------------------------------ data access
    @property
    def customers(self) -> pd.DataFrame:
        if self._customers is None:
            self._customers = pd.read_csv(self._customers_file)
        return self._customers

    def get_customer(self, customer_id: int):
        row = self.customers[self.customers.customer_id == customer_id]
        return None if row.empty else row.iloc[0].to_dict()

    # ------------------------------------------------------------------ core scoring
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the SAME row-level cleaning rules as in training."""
        df = df.copy()
        for c in ("login_count_30d", "days_since_last_login"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return apply_row_rules(df, self.params)

    def band(self, p: float) -> str:
        return "High" if p >= self.t_high else "Medium" if p >= self.t_medium else "Low"

    def score_df(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Vectorised scoring of a table of customers (raw inputs, cleaning applied here)."""
        df = self.prepare(raw)
        p = self.pipe.predict_proba(df)[:, 1]
        F = self.fb.transform(df)
        cl = self.segmenter.predict(F)
        clv = C.clv_proxy(df["balance"].to_numpy(float), df["products_number"].to_numpy(float))
        ev = C.expected_value_of_contact(p, clv)
        out = pd.DataFrame({
            "churn_probability": p, "risk_band": [self.band(x) for x in p],
            "segment": cl, "persona": [self.cluster_to_persona[c] for c in cl],
            "clv_eur": clv, "expected_value_of_contact_eur": ev, "contact_recommended": ev > 0,
        }, index=raw.index)
        return out

    # ------------------------------------------------------------------ explanation
    def explain(self, record_df: pd.DataFrame, top=5):
        """Model-agnostic factor impacts (two-point approximation of Shapley values) for groups of inputs.

        For each group g: impact = 0.5 * [ (p(customer) - p(customer with g set to typical))
                                         + (p(typical with g set to customer) - p(typical)) ]
        i.e. the average of "remove the factor from this customer" and "add the factor to a typical customer".
        Using both directions avoids the saturation problem of one-at-a-time removal for very high-risk customers.
        Positive = raises the churn probability."""
        df = self.prepare(record_df)
        typical = pd.DataFrame([{**self.ref}])
        typical["login_missing_flag"] = 0
        typical = typical[df.columns.intersection(typical.columns)]
        for c in df.columns:                       # same column set / order as the customer frame
            if c not in typical:
                typical[c] = df[c].iloc[0]
        typical = typical[df.columns]
        p0 = float(self.pipe.predict_proba(df)[:, 1][0])
        pt = float(self.pipe.predict_proba(typical)[:, 1][0])
        rows = []
        for name, cols in GROUPS.items():
            out_, in_ = df.copy(), typical.copy()
            for c in cols:
                out_[c] = typical[c].iloc[0]
                in_[c] = df[c].iloc[0]
            p_out = float(self.pipe.predict_proba(out_)[:, 1][0])
            p_in = float(self.pipe.predict_proba(in_)[:, 1][0])
            impact = 0.5 * ((p0 - p_out) + (p_in - pt))
            rows.append({"factor": name, "impact": impact, "customer_value": self._describe_value(name, df.iloc[0]),
                         "typical_value": self._describe_value(name, pd.Series(self.ref))})
        rows.sort(key=lambda r: -abs(r["impact"]))
        return rows[:top], rows

    @staticmethod
    def _describe_value(name, r):
        cnt = [float(r[c]) for c in C.TXN_COUNT_COLS]
        if name.startswith("Transaction activity"):
            ratio = (np.mean(cnt[3:]) + 1) / (np.mean(cnt[:3]) + 1)
            return f"{sum(cnt):.0f} transactions in 6 months, recent/early ratio {ratio:.2f}"
        if name.startswith("Spending"):
            return f"EUR {sum(float(r[c]) for c in C.TXN_AMOUNT_COLS):,.0f} in 6 months"
        if name == "Time since last transaction": return f"{float(r['days_since_last_txn']):.0f} days"
        if name.startswith("App usage"): return f"{float(r['login_count_30d']):.0f} logins in 30 days, last {float(r['days_since_last_login']):.0f} days ago"
        if name == "Complaints": return f"{float(r['complaints_6m']):.0f} in 6 months"
        if name == "Channel mix": return f"{float(r['mobile_share']):.0%} mobile, preferred: {r['preferred_channel']}"
        if name == "Age": return f"{float(r['age']):.0f} years"
        if name == "Country": return str(r["country"])
        if name == "Gender": return str(r["gender"])
        if name == "Number of products": return f"{float(r['products_number']):.0f}"
        if name.startswith("Active"): return "active" if float(r["active_member"]) == 1 else "inactive"
        if name == "Account balance": return f"EUR {float(r['balance']):,.0f}"
        if name == "Estimated salary": return f"EUR {float(r['estimated_salary']):,.0f}"
        if name == "Credit score": return f"{float(r['credit_score']):.0f}"
        if name == "Tenure": return f"{float(r['tenure']):.0f} years"
        if name == "Credit card": return "yes" if float(r["credit_card"]) == 1 else "no"
        return ""

    # ------------------------------------------------------------------ single customer
    def score_record(self, record: dict, explain=True, log=True) -> dict:
        df = pd.DataFrame([record])
        s = self.score_df(df).iloc[0]
        persona = s["persona"]; band = s["risk_band"]
        out = {
            "churn_probability": round(float(s["churn_probability"]), 4),
            "risk_band": band,
            "segment_id": int(s["segment"]), "persona": persona,
            "persona_description": self.personas[persona]["description"],
            "clv_eur": round(float(s["clv_eur"]), 2),
            "expected_value_of_contact_eur": round(float(s["expected_value_of_contact_eur"]), 2),
            "contact_recommended": bool(s["contact_recommended"]),
            "recommended_action": self.personas[persona]["actions"][band.lower()],
            "model_version": self.version, "threshold": self.thr,
        }
        if explain:
            top, _ = self.explain(df)
            out["top_factors"] = [{"factor": r["factor"], "impact_pp": round(100 * r["impact"], 1),
                                   "direction": "raises risk" if r["impact"] > 0 else "lowers risk",
                                   "customer_value": r["customer_value"], "typical_value": r["typical_value"]} for r in top]
        if log and self.log_predictions:
            self._log(record, out)
        return out

    def _log(self, record, out):
        try:
            C.LOGS.mkdir(exist_ok=True)
            entry = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model_version": self.version,
                     "input": record, "churn_probability": out["churn_probability"], "risk_band": out["risk_band"], "persona": out["persona"]}
            with open(self.log_path, "a") as f:
                f.write(json.dumps(entry, default=float) + "\n")
        except Exception:      # read-only file systems on some free tiers must not break scoring
            pass

    def lookup(self, customer_id: int, explain=True):
        rec = self.get_customer(customer_id)
        if rec is None:
            return None
        rec = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in rec.items()}
        out = self.score_record({k: rec[k] for k in C.INPUT_COLS + ["login_missing_flag"] if k in rec}, explain=explain, log=False)
        out["customer_id"] = int(customer_id)
        out["actual_churn_label"] = int(rec["churn"])
        return out
