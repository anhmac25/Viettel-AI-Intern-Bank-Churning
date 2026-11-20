"""Model monitoring (Module 7): drift measures, trigger rules and batch evaluation."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# ---------------------------------------------------------------- trigger configuration
TRIGGERS = {
    "psi_warning": 0.10,          # PSI between 0.10 and 0.25: moderate shift
    "psi_alert": 0.25,            # PSI above 0.25: significant shift
    "drifted_share_alert": 0.30,  # alert if more than 30% of monitored features have PSI > psi_alert
    "prediction_shift_rel": 0.20, # relative change of the mean predicted probability vs reference
    "auc_drop": 0.05,             # absolute AUC drop vs baseline
    "recall_target": 0.85,        # recall at the business threshold must stay above this
    "consecutive_breaches": 2,    # performance breaches needed to trigger retraining
    "missing_rate": 0.05,         # data-quality: max share of missing values in any field
    "scheduled_review_months": 3, # retrain at least every quarter
}
MONITORED_NUM = ["txn_freq_6m", "txn_amount_6m", "txn_ratio_recent", "txn_slope", "login_count_30d", "recency_days",
                 "complaints_6m", "mobile_share", "age", "balance", "products_number", "active_member", "credit_score",
                 "estimated_salary"]
MONITORED_CAT = ["country", "gender", "preferred_channel"]


# ---------------------------------------------------------------- drift measures
def psi(ref, cur, bins=10, eps=1e-4) -> float:
    """Population Stability Index. Numeric: reference-decile bins; low-cardinality / categorical: category shares."""
    ref, cur = pd.Series(ref).dropna(), pd.Series(cur).dropna()
    if ref.dtype == object or str(ref.dtype) == "str" or ref.nunique() <= 10:
        cats = sorted(set(ref.unique()) | set(cur.unique()), key=str)
        r = ref.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
        c = cur.value_counts(normalize=True).reindex(cats).fillna(0).to_numpy()
    else:
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        r = np.histogram(ref, edges)[0] / len(ref)
        c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, eps, None), np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def ks_pvalue(ref, cur) -> float:
    return float(stats.ks_2samp(pd.Series(ref).dropna(), pd.Series(cur).dropna()).pvalue)


def feature_drift_table(F_ref: pd.DataFrame, F_cur: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in MONITORED_NUM + MONITORED_CAT:
        v = psi(F_ref[c], F_cur[c])
        rows.append({"feature": c, "psi": v,
                     "ks_pvalue": ks_pvalue(F_ref[c], F_cur[c]) if c in MONITORED_NUM else np.nan,
                     "status": "ALERT" if v > TRIGGERS["psi_alert"] else "WARNING" if v > TRIGGERS["psi_warning"] else "OK"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- performance
def performance(y, p, thr) -> dict:
    pred = (p >= thr).astype(int)
    return {"auc": float(roc_auc_score(y, p)), "recall": float(recall_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "churn_rate": float(np.mean(y)), "mean_prediction": float(np.mean(p)), "share_flagged": float(pred.mean())}


# ---------------------------------------------------------------- trigger evaluation
def evaluate_batch(drift: pd.DataFrame, top_features, p_ref_mean, perf, missing_rate, schema_ok, baseline_auc, history) -> dict:
    """Returns status per signal and the recommended action. `history` = list of earlier batch results."""
    T = TRIGGERS
    top = drift[drift.feature.isin(top_features)]
    share_alert = float((drift.psi > T["psi_alert"]).mean())
    if (top.psi > T["psi_alert"]).any() or share_alert > T["drifted_share_alert"]:
        s_input = "ALERT"
    elif (drift.psi > T["psi_warning"]).any() and (top.psi > T["psi_warning"]).any():
        s_input = "WARNING"
    else:
        s_input = "OK"

    rel = abs(perf["mean_prediction"] - p_ref_mean) / p_ref_mean
    s_pred = "WARNING" if rel > T["prediction_shift_rel"] else "OK"

    breach = (baseline_auc - perf["auc"] > T["auc_drop"]) or (perf["recall"] < T["recall_target"])
    # batches blocked for data quality are not valid performance observations: they are skipped in the chain
    valid = [h for h in history if h.get("data_quality") != "BLOCK"]
    prev_breach = bool(valid) and valid[-1].get("perf_breach", False)
    if breach and prev_breach:
        s_perf = "ALERT"
    elif breach:
        s_perf = "WARNING"
    else:
        s_perf = "OK"

    s_quality = "BLOCK" if (missing_rate > T["missing_rate"] or not schema_ok) else "OK"

    if s_quality == "BLOCK":
        action = "Block scoring of this batch and alert the data owner"
    elif s_perf == "ALERT":
        action = "RETRAIN and revalidate before replacing the model"
    elif s_input == "ALERT":
        action = "Schedule retraining and investigate the data source"
    elif "WARNING" in (s_input, s_pred, s_perf):
        action = "Investigate (data change, segment mix, labels)"
    else:
        action = "No action"
    return {"input_drift": s_input, "prediction_drift": s_pred, "performance": s_perf, "data_quality": s_quality,
            "perf_breach": bool(breach), "prediction_shift_rel": float(rel), "share_features_alert": share_alert,
            "max_psi_top": float(top.psi.max()), "action": action}
