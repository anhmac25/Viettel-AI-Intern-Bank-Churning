"""Module 7 - Model monitoring.

There is no real production traffic in this project, so six monthly batches are SIMULATED, each a random
draw of 600 customers from the held-out TEST set, with controlled problems injected:

  month 1  stable                      (should stay quiet)
  month 2  mild shift                  (-10% activity, +5 pts mobile share)
  month 3  strong covariate drift      (app outage + activity drop: fewer transactions/logins, more complaints)
  month 4  concept drift               (competitor offer: low-risk personas suddenly churn; features look normal)
  month 5  data-quality failure        (9% of login data missing)
  month 6  concept drift continues     (second consecutive performance breach -> retraining trigger)

Outputs (reports/monitoring/): summary.json, monitoring_summary.csv, feature_drift_by_month.csv,
evidently_month_{1,3,4}.html, and figures.
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.presets import ClassificationPreset, DataDriftPreset

from churnlib import config as C
from churnlib import monitoring as M
from churnlib.service import ChurnService

svc = ChurnService(log_predictions=False)
rng = np.random.default_rng(2026)
train = pd.read_csv(C.TRAIN_FILE); test = pd.read_csv(C.TEST_FILE)
thr, baseline_auc = svc.thr, svc.meta["metrics_test_at_business_threshold"]["roc_auc"]

# --- reference = training data
F_ref = svc.fb.transform(svc.prepare(train))
p_ref = svc.pipe.predict_proba(svc.prepare(train))[:, 1]
p_ref_mean = float(p_ref.mean())
gain = pd.Series(svc.pipe.named_steps["clf"].feature_importances_,
                 index=svc.pipe.named_steps["prep"].get_feature_names_out())
top_features = [f for f in gain.sort_values(ascending=False).index if f in M.MONITORED_NUM][:5]
print("top-5 monitored features by importance:", top_features)

# ------------------------------------------------------------------ scenario generators (act on RAW inputs)
cnt, amt = C.TXN_COUNT_COLS, C.TXN_AMOUNT_COLS


def thin(df, factor):
    for c in cnt: df[c] = rng.binomial(df[c].astype(int), factor)
    for c in amt: df[c] = (df[c] * factor).round(2)


def s_stable(df): return df
def s_mild(df):
    thin(df, 0.9)
    df["mobile_share"] = (df.mobile_share + 0.05).clip(upper=0.95)
    rest = (1 - df.mobile_share) / (df.atm_share + df.branch_share).replace(0, 1)
    df["atm_share"], df["branch_share"] = df.atm_share * rest, df.branch_share * rest
    return df
def s_outage(df):
    thin(df, 0.62)
    df["login_count_30d"] = rng.binomial(df.login_count_30d.fillna(8).astype(int), 0.55).astype(float)
    df["days_since_last_login"] = (df.days_since_last_login.fillna(5) * 1.8).round()
    df["days_since_last_txn"] = (df.days_since_last_txn * 1.5).round().clip(upper=180).astype(int)
    df["complaints_6m"] = df.complaints_6m + rng.poisson(0.8, len(df))
    return df
def s_quality(df):
    m = rng.random(len(df)) < 0.09
    df.loc[m, ["login_count_30d", "days_since_last_login"]] = np.nan
    return df


def concept_shift(df, personas, rate=0.35):
    """Labels change, inputs look normal: customers of two low-risk personas churn to a competitor."""
    sc = svc.score_df(df)
    hit = sc.persona.isin(personas).to_numpy() & (rng.random(len(df)) < rate)
    df["churn"] = np.where(hit, 1, df["churn"])
    return df


LOW_RISK = ["Digital Power Users", "High-Balance Savers"]
SCENARIOS = [
    ("Month 1", "Stable", s_stable), ("Month 2", "Mild shift", s_mild), ("Month 3", "App outage / activity drop", s_outage),
    ("Month 4", "Competitor offer (concept drift)", lambda d: concept_shift(d, LOW_RISK)),
    ("Month 5", "Login tracking failure", s_quality),
    ("Month 6", "Competitor offer continues", lambda d: concept_shift(d, LOW_RISK)),
]
# the clean test table has imputed logins: restore NaN where they were originally missing, like a raw stream
test_raw = test.copy()
test_raw.loc[test_raw.login_missing_flag == 1, ["login_count_30d", "days_since_last_login"]] = np.nan

history, summary, drift_rows = [], [], []
for i, (name, desc, fn) in enumerate(SCENARIOS, 1):
    batch = test_raw.sample(600, random_state=100 + i).copy().reset_index(drop=True)
    batch = fn(batch)
    missing = float(batch[["login_count_30d", "days_since_last_login"]].isna().any(axis=1).mean())
    schema_ok = all(c in batch.columns for c in C.INPUT_COLS)
    prepared = svc.prepare(batch)
    p = svc.pipe.predict_proba(prepared)[:, 1]
    F_cur = svc.fb.transform(prepared)
    drift = M.feature_drift_table(F_ref, F_cur)
    perf = M.performance(batch.churn.to_numpy(), p, thr)
    res = M.evaluate_batch(drift, top_features, p_ref_mean, perf, missing, schema_ok, baseline_auc, history)
    history.append(res)
    rec = {"month": name, "scenario": desc, "n": len(batch), "missing_rate": round(missing, 4), **{k: round(v, 4) for k, v in perf.items()}, **res}
    summary.append(rec)
    drift_rows += [{"month": name, **r} for r in drift.to_dict("records")]
    print(f"{name} | {desc:34s} | AUC {perf['auc']:.3f} recall {perf['recall']:.3f} | input {res['input_drift']:7s} pred {res['prediction_drift']:7s} "
          f"perf {res['performance']:7s} quality {res['data_quality']:5s} -> {res['action']}")

    if i in (1, 3, 4):      # Evidently HTML reports (3-4 MB each, so only for the three illustrative months)
        cols = M.MONITORED_NUM + M.MONITORED_CAT
        mk = lambda F, df_, pp: Dataset.from_pandas(
            pd.concat([F[cols].reset_index(drop=True), pd.DataFrame({"churn": df_.churn.to_numpy(), "churn_probability": pp,
                                                                    "predicted": (pp >= thr).astype(int)})], axis=1),
            data_definition=DataDefinition(numerical_columns=M.MONITORED_NUM, categorical_columns=M.MONITORED_CAT,
                                           classification=[BinaryClassification(target="churn", prediction_probas="churn_probability",
                                                                               prediction_labels="predicted", pos_label=1)]))
        ref_n = 2000
        idx = np.random.default_rng(1).choice(len(train), ref_n, replace=False)
        ref_ds = mk(F_ref.iloc[idx], train.iloc[idx], p_ref[idx])
        cur_ds = mk(F_cur, batch, p)
        rep = Report([DataDriftPreset(columns=cols), ClassificationPreset()], include_tests=True)
        rep.run(cur_ds, ref_ds).save_html(str(C.MONITOR / f"evidently_month_{i}.html"))

# ------------------------------------------------------------------ persist
sm = pd.DataFrame(summary)
sm.to_csv(C.MONITOR / "monitoring_summary.csv", index=False)
pd.DataFrame(drift_rows).to_csv(C.MONITOR / "feature_drift_by_month.csv", index=False)
(C.MONITOR / "summary.json").write_text(json.dumps({
    "reference": {"rows": len(train), "mean_prediction": p_ref_mean, "baseline_auc": baseline_auc, "threshold": thr,
                  "top_features": top_features}, "triggers": M.TRIGGERS, "batches": summary}, indent=2, default=float))

# ------------------------------------------------------------------ figures
dd = pd.DataFrame(drift_rows).pivot(index="feature", columns="month", values="psi")
fig, ax = plt.subplots(figsize=(9, 6.5))
sns.heatmap(dd.loc[M.MONITORED_NUM + M.MONITORED_CAT], annot=True, fmt=".2f", cmap="YlOrRd", vmin=0, vmax=0.6, ax=ax, cbar_kws={"label": "PSI"})
ax.set_title("Population Stability Index by feature and month (0.10 = warning, 0.25 = alert)"); ax.set_ylabel("")
fig.tight_layout(); fig.savefig(C.FIGS / "monitor_psi_heatmap.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
x = sm.month
axes[0].plot(x, sm.auc, "o-", color="#4C78A8"); axes[0].axhline(baseline_auc, color="grey", ls=":", label="baseline")
axes[0].axhline(baseline_auc - M.TRIGGERS["auc_drop"], color="#E4572E", ls="--", label="retrain trigger"); axes[0].set_title("AUC-ROC"); axes[0].legend(); axes[0].set_ylim(0.6, 1)
axes[1].plot(x, sm.recall, "o-", color="#4C78A8"); axes[1].axhline(M.TRIGGERS["recall_target"], color="#E4572E", ls="--", label="target"); axes[1].set_title(f"Recall at threshold {thr:.2f}"); axes[1].legend(); axes[1].set_ylim(0.4, 1)
axes[2].plot(x, sm.mean_prediction, "o-", color="#4C78A8"); axes[2].axhline(p_ref_mean, color="grey", ls=":", label="reference")
axes[2].axhline(p_ref_mean * 1.2, color="#E4572E", ls="--", label="+/-20%"); axes[2].axhline(p_ref_mean * 0.8, color="#E4572E", ls="--"); axes[2].set_title("Mean predicted churn probability"); axes[2].legend()
for a in axes: a.tick_params(axis="x", rotation=30)
fig.tight_layout(); fig.savefig(C.FIGS / "monitor_performance.png", dpi=130); plt.close(fig)

md = ["# Monitoring results (simulated months)", "", f"Reference: {len(train)} training customers; baseline test AUC {baseline_auc:.3f}; threshold {thr:.2f}.", "",
      "| month | scenario | AUC | recall | input drift | prediction drift | performance | data quality | action |", "|---|---|---|---|---|---|---|---|---|"]
md += [f"| {r['month']} | {r['scenario']} | {r['auc']:.3f} | {r['recall']:.3f} | {r['input_drift']} | {r['prediction_drift']} | {r['performance']} | {r['data_quality']} | {r['action']} |" for r in summary]
(C.REPORTS / "monitoring_results.md").write_text("\n".join(md), encoding="utf-8")
