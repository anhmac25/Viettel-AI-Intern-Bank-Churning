"""Customer segmentation (Module 5 / Branch B): K-Means on behavioural/RFM features + personas."""
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["recency_days", "txn_freq_6m", "txn_amount_6m", "login_count_30d", "mobile_share",
                "balance", "products_number", "txn_ratio_recent", "age"]
LOG_COLS = ["txn_amount_6m", "balance"]
LABELS = {
    "recency_days": "days since last transaction", "txn_freq_6m": "transaction frequency", "txn_amount_6m": "spending",
    "login_count_30d": "app logins", "mobile_share": "mobile-app usage", "balance": "balance",
    "products_number": "number of products", "txn_ratio_recent": "recent activity trend", "age": "age",
}


def _matrix(F: pd.DataFrame) -> np.ndarray:
    X = F[SEG_FEATURES].astype(float).copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c])
    return X.to_numpy()


class Segmenter(BaseEstimator):
    def __init__(self, k=4, random_state=42):
        self.k = k
        self.random_state = random_state

    def fit(self, F: pd.DataFrame):
        self.scaler_ = StandardScaler().fit(_matrix(F))
        self.km_ = KMeans(self.k, n_init=20, random_state=self.random_state).fit(self.scaler_.transform(_matrix(F)))
        return self

    def predict(self, F: pd.DataFrame) -> np.ndarray:
        return self.km_.predict(self.scaler_.transform(_matrix(F)))

    def profile(self, F: pd.DataFrame, labels=None) -> pd.DataFrame:
        labels = self.predict(F) if labels is None else labels
        return F[SEG_FEATURES].assign(cluster=labels).groupby("cluster").mean()

    def zscores(self, F: pd.DataFrame, labels=None) -> pd.DataFrame:
        prof = self.profile(F, labels)
        return (prof - F[SEG_FEATURES].mean()) / F[SEG_FEATURES].std()


def assign_personas(z: pd.DataFrame, templates: dict) -> dict:
    """One-to-one assignment of clusters to persona templates (Hungarian algorithm).

    templates: {persona: {feature: target_z}} - features not listed are ignored.
    Returns {cluster_id: persona}. Deterministic given the clusters."""
    names = list(templates)
    cost = np.zeros((len(z), len(names)))
    for i, cl in enumerate(z.index):
        for j, nm in enumerate(names):
            t = templates[nm]
            cost[i, j] = np.mean([(z.loc[cl, f] - v) ** 2 for f, v in t.items()])
    r, c = linear_sum_assignment(cost)
    return {int(z.index[i]): names[j] for i, j in zip(r, c)}


def describe(z_row: pd.Series, thr=0.35) -> str:
    """Plain-language description from z-scores (what is clearly higher / lower than average)."""
    hi = [LABELS[f] for f, v in z_row.sort_values(ascending=False).items() if v >= thr][:3]
    lo = [LABELS[f] for f, v in z_row.sort_values().items() if v <= -thr][:3]
    parts = []
    if hi: parts.append("higher " + ", ".join(hi))
    if lo: parts.append("lower " + ", ".join(lo))
    return "; ".join(parts) if parts else "close to the average customer on all dimensions"
