"""Module 8 - Business translation. Computes the business impact per segment on the held-out test set and
collects every number needed for the written report.

Outputs: reports/business_impact.md, reports/RESULTS_FOR_REPORT.md, docs/monitoring_triggers.md
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from churnlib import config as C
from churnlib import monitoring as M
from churnlib.modeling import realized_net_benefit
from churnlib.service import ChurnService

svc = ChurnService(log_predictions=False)
test = pd.read_csv(C.TEST_FILE)
sc = svc.score_df(test)
y = test.churn.to_numpy()
flag = (sc.churn_probability >= svc.thr).to_numpy()
clv = sc.clv_eur.to_numpy()
meta, personas = svc.meta, svc.personas

# ------------------------------------------------------------------ impact by persona (test set)
rows = []
for p in sorted(personas, key=lambda x: -personas[x]["churn_rate_all"]):
    m = (sc.persona == p).to_numpy()
    rows.append({"persona": p, "customers": int(m.sum()), "churn rate %": round(100 * y[m].mean(), 1),
                 "share flagged %": round(100 * flag[m].mean(), 1),
                 "churners reached %": round(100 * (flag[m] & (y[m] == 1)).sum() / max((y[m] == 1).sum(), 1), 1),
                 "avg CLV (EUR)": round(clv[m].mean()),
                 "net benefit of contacting flagged (EUR)": round(realized_net_benefit(y[m], flag[m], clv[m]))})
impact = pd.DataFrame(rows)
tot = realized_net_benefit(y, flag, clv)

bands = pd.DataFrame({"band": sc.risk_band, "churn": y}).groupby("band").churn.agg(["count", "mean"]).reindex(["High", "Medium", "Low"])
bands["mean"] = (bands["mean"] * 100).round(1)
bands.columns = ["customers", "observed churn rate %"]
by_ev = float(realized_net_benefit(y, sc.contact_recommended.to_numpy(), clv))

b = C.BUSINESS
md = ["# Business impact (held-out test set)", "",
      f"Assumptions (replace with the bank's figures): {b}. Test customers: {len(test)}.", "",
      "## Observed churn by risk band (the bands are well calibrated)", "", bands.to_markdown(), "",
      "## Impact by segment when contacting every customer with churn probability >= "
      f"{svc.thr:.2f}", "", impact.to_markdown(index=False), "",
      f"Total net benefit of the model-based policy on the test set: **EUR {tot:,.0f}**. "
      f"Policy 'contact when expected value of contact > 0 for that customer' gives EUR {by_ev:,.0f}.", "",
      "Note: the euro figures depend entirely on the assumed CLV, incentive, contact cost and acceptance rate, and on the synthetic behaviour; "
      "they illustrate the method, they are not a forecast."]
(C.REPORTS / "business_impact.md").write_text("\n".join(md), encoding="utf-8")

# ------------------------------------------------------------------ monitoring triggers doc
T = M.TRIGGERS
trig = ["# Monitoring signals and retraining triggers", "",
        "| Signal | Metric | Warning | Alert / action |", "|---|---|---|---|",
        f"| Input data drift | PSI per monitored feature (and KS test); top-5 features by model importance | PSI > {T['psi_warning']} on a top feature | PSI > {T['psi_alert']} on any top-5 feature, or > {T['drifted_share_alert']:.0%} of monitored features above {T['psi_alert']} -> schedule retraining, investigate source |",
        f"| Prediction drift | relative change of the mean predicted probability vs reference | > {T['prediction_shift_rel']:.0%} | investigate data change and segment mix |",
        f"| Model performance | AUC-ROC and recall at the business threshold (labels arrive after the prediction horizon) | AUC drop > {T['auc_drop']} vs baseline, or recall < {T['recall_target']:.0%}, in one month | the same breach in {T['consecutive_breaches']} consecutive valid months -> RETRAIN and revalidate |",
        f"| Data quality | missing-value rate per field; schema check | - | missing > {T['missing_rate']:.0%} or schema violation -> BLOCK the batch and alert the data owner (blocked batches are skipped in the consecutive-breach rule) |",
        f"| Scheduled review | calendar | - | retrain at least every {T['scheduled_review_months']} months (the prediction horizon is one quarter) |", "",
        "PSI rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift. Thresholds are starting points and must be recalibrated on real data.", "",
        "Why performance monitoring is required in addition to drift monitoring: in the simulation (reports/monitoring_results.md) the competitor-offer "
        "months change who churns without changing the inputs, so PSI stays near zero while AUC falls from about 0.93 to about 0.74."]
(C.DOCS / "monitoring_triggers.md").write_text("\n".join(trig), encoding="utf-8")

# ------------------------------------------------------------------ numbers for the written report
read = lambda f: (C.REPORTS / f).read_text(encoding="utf-8")
mc = pd.read_csv(C.REPORTS / "model_comparison.csv")
imb = pd.read_csv(C.REPORTS / "imbalance_comparison.csv")
mon = pd.read_csv(C.MONITOR / "monitoring_summary.csv")[["month", "scenario", "auc", "recall", "input_drift", "prediction_drift", "performance", "data_quality", "action"]]
mf, m5 = meta["metrics_test_at_business_threshold"], meta["metrics_test_at_0_5"]
allc = pd.read_csv(C.MERGED_CLEAN)
out = ["# Numbers for the written report", "",
       "Every figure below was produced by the pipeline; copy them into the yellow placeholders of the Word report.", "",
       "## Dataset", "",
       f"- customers: {len(allc):,}; columns after merge: {allc.shape[1]}; overall churn rate: {allc.churn.mean():.2%}",
       f"- train / test: {meta['train_rows']} / {meta['test_rows']} (stratified 80/20, seed {C.SEED}); train churn rate {meta['train_churn_rate']:.2%}",
       f"- zero-balance customers: {(allc.balance == 0).mean():.1%}", "",
       "## EDA (Table 3.4 of the report)", "", "See reports/eda_summary.md - churn rates by country, age band, products, activity, gender.", "",
       "## Cleaning log (Table 3.5)", "", pd.read_csv(C.REPORTS / "cleaning_log.csv").to_markdown(index=False), "",
       "## Class-imbalance experiment (5-fold CV on train)", "", imb.to_markdown(index=False), "",
       f"Chosen strategy: {meta['imbalance_strategy']} for the final model. SMOTE and class weights did not improve ROC-AUC enough to justify "
       "distorting the predicted probabilities.", "",
       "## Model comparison (Table 3.7)", "", mc.to_markdown(index=False), "",
       f"- final model: **{meta['algorithm']}**, CV ROC-AUC {meta['cv_roc_auc']:.4f}",
       f"- at business threshold {meta['threshold']:.2f}: ROC-AUC {mf['roc_auc']:.3f}, PR-AUC {mf['pr_auc']:.3f}, precision {mf['precision']:.3f}, recall {mf['recall']:.3f}, F1 {mf['f1']:.3f}",
       f"- at threshold 0.5: precision {m5['precision']:.3f}, recall {m5['recall']:.3f}, F1 {m5['f1']:.3f}",
       f"- profile-only AUC {meta['auc_profile_only']:.3f} vs profile + behaviour {meta['auc_profile_plus_behaviour']:.3f}",
       f"- out-of-fold Brier score {meta['brier_oof']:.4f}", "",
       "## Segments (Table 3.8) and segment impact", "", impact.to_markdown(index=False), "",
       "## Observed churn by risk band", "", bands.to_markdown(), "",
       "## Monitoring (Table 3.10 and the simulation)", "", mon.to_markdown(index=False), "",
       "## Where the other results live", "",
       "- segmentation details and the segment-as-feature test: reports/segmentation_results.md",
       "- model details and strategy comparison: reports/model_results.md",
       "- business impact by segment: reports/business_impact.md",
       "- deployed URLs: fill in after deployment (README, section 'Deploy')."]
(C.REPORTS / "RESULTS_FOR_REPORT.md").write_text("\n".join(out), encoding="utf-8")
print(impact.to_string(index=False)); print(bands); print("total net benefit", round(tot))
