"""Feature engineering (Module 4). One scikit-learn transformer used in training AND serving, so the
application computes exactly the same features as the training pipeline."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import TXN_AMOUNT_COLS, TXN_COUNT_COLS

_X = np.arange(1, 7, dtype=float)
_XC = _X - _X.mean()


def _slope(mat: np.ndarray) -> np.ndarray:
    """OLS slope of each row against month index 1..6 (vectorised)."""
    return (mat * _XC).sum(axis=1) / (_XC ** 2).sum()


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Turns a cleaned customer table into model features.

    fit() learns the quintile edges used for the RFM scores from the TRAINING data only.
    """

    NUMERIC = [
        "credit_score", "age", "tenure", "balance", "products_number", "credit_card", "active_member",
        "estimated_salary", "txn_freq_6m", "txn_amount_6m", "avg_ticket", "recency_days", "txn_ratio_recent",
        "txn_slope", "amount_ratio_recent", "login_count_30d", "days_since_last_login", "login_missing_flag",
        "mobile_share", "branch_share", "atm_share", "complaints_6m", "complaint_flag", "balance_to_salary",
        "products_per_tenure", "zero_balance_flag", "R_score", "F_score", "M_score", "rfm_score",
    ]
    CATEGORICAL = ["country", "gender", "preferred_channel"]

    def fit(self, X, y=None):
        f = self._base_features(X)
        self.edges_ = {
            "recency": np.quantile(f["recency_days"], [0.2, 0.4, 0.6, 0.8]),
            "freq": np.quantile(f["txn_freq_6m"], [0.2, 0.4, 0.6, 0.8]),
            "monetary": np.quantile(f["txn_amount_6m"], [0.2, 0.4, 0.6, 0.8]),
        }
        return self

    # ---------------------------------------------------------------------------------
    def _base_features(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.reset_index(drop=True)
        cnt = X[TXN_COUNT_COLS].to_numpy(float)
        amt = X[TXN_AMOUNT_COLS].to_numpy(float)
        f = pd.DataFrame(index=X.index)
        for c in ["credit_score", "age", "tenure", "balance", "products_number", "credit_card",
                  "active_member", "estimated_salary", "login_count_30d", "days_since_last_login",
                  "mobile_share", "branch_share", "atm_share", "complaints_6m"]:
            f[c] = X[c].astype(float)
        f["login_missing_flag"] = X["login_missing_flag"].astype(float) if "login_missing_flag" in X else 0.0
        f["txn_freq_6m"] = cnt.sum(axis=1)
        f["txn_amount_6m"] = amt.sum(axis=1)
        f["avg_ticket"] = f["txn_amount_6m"] / np.maximum(f["txn_freq_6m"], 1)
        f["recency_days"] = X["days_since_last_txn"].astype(float)
        f["txn_ratio_recent"] = (cnt[:, 3:].mean(axis=1) + 1) / (cnt[:, :3].mean(axis=1) + 1)
        f["amount_ratio_recent"] = (amt[:, 3:].mean(axis=1) + 1) / (amt[:, :3].mean(axis=1) + 1)
        f["txn_slope"] = _slope(cnt) / (cnt.mean(axis=1) + 1)
        f["complaint_flag"] = (f["complaints_6m"] >= 1).astype(float)
        f["balance_to_salary"] = f["balance"] / (f["estimated_salary"] + 1)
        f["products_per_tenure"] = f["products_number"] / (f["tenure"] + 1)
        f["zero_balance_flag"] = (f["balance"] == 0).astype(float)
        for c in self.CATEGORICAL:
            f[c] = X[c].astype(str).values
        return f

    def transform(self, X):
        f = self._base_features(X)
        # quintile scores 1..5 using edges fitted on the training data (recency reversed)
        f["R_score"] = 5 - np.searchsorted(self.edges_["recency"], f["recency_days"], side="right")
        f["F_score"] = 1 + np.searchsorted(self.edges_["freq"], f["txn_freq_6m"], side="right")
        f["M_score"] = 1 + np.searchsorted(self.edges_["monetary"], f["txn_amount_6m"], side="right")
        f["R_score"] = f["R_score"].clip(1, 5).astype(float)
        f["F_score"] = f["F_score"].astype(float)
        f["M_score"] = f["M_score"].astype(float)
        f["rfm_score"] = f["R_score"] + f["F_score"] + f["M_score"]
        return f[self.NUMERIC + self.CATEGORICAL]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.NUMERIC + self.CATEGORICAL)
