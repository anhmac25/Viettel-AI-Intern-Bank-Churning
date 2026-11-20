"""Pipelines and evaluation helpers shared by the training scripts."""
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config as C
from .features import FeatureBuilder


def make_preprocessor(numeric, categorical):
    return ColumnTransformer(
        [("num", StandardScaler(), list(numeric)),
         ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), list(categorical))],
        verbose_feature_names_out=False,
    )


def make_model(name, imbalance="none", pos_weight=1.0, **params):
    cw = "balanced" if imbalance == "class_weight" else None
    if name == "logreg":
        return LogisticRegression(max_iter=2000, class_weight=cw, **params)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, class_weight=cw, n_jobs=-1, random_state=C.SEED, **params)
    if name == "xgb":
        spw = pos_weight if imbalance == "class_weight" else 1.0
        kw = dict(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                  eval_metric="logloss", random_state=C.SEED, n_jobs=1, scale_pos_weight=spw)
        kw.update(params)
        return XGBClassifier(**kw)
    raise ValueError(name)


def make_pipeline(name, numeric, categorical, imbalance="none", pos_weight=1.0, **params):
    steps = [("features", FeatureBuilder()), ("prep", make_preprocessor(numeric, categorical))]
    if imbalance == "smote":
        steps.append(("smote", SMOTE(random_state=C.SEED)))
    steps.append(("clf", make_model(name, imbalance, pos_weight, **params)))
    return ImbPipeline(steps)


def metrics_at(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p),
        "precision": precision_score(y, pred, zero_division=0), "recall": recall_score(y, pred),
        "f1": f1_score(y, pred), "threshold": thr,
    }


# ---------------------------------------------------------------- business value
def realized_net_benefit(y, flagged, clv):
    """Realised net benefit (EUR) of contacting the flagged customers.
    churner contacted: a*(CLV - incentive) - cost ; retained contacted: -a*incentive - cost."""
    b = C.BUSINESS
    y = np.asarray(y); flagged = np.asarray(flagged).astype(bool); clv = np.asarray(clv)
    gain = np.where(y == 1, b["accept_rate"] * (clv - b["incentive"]) - b["contact_cost"],
                    -b["accept_rate"] * b["incentive"] - b["contact_cost"])
    return float((gain * flagged).sum())


def best_threshold(y, p, clv, grid=None):
    grid = np.linspace(0.02, 0.9, 89) if grid is None else grid
    vals = np.array([realized_net_benefit(y, p >= t, clv) for t in grid])
    return float(grid[vals.argmax()]), pd.DataFrame({"threshold": grid, "net_benefit": vals})
