"""Module 5 / Branch A - churn classification.

Logistic regression, random forest and XGBoost are tuned with randomised search (5-fold stratified CV
on the TRAIN set), compared on the untouched TEST set, and the final model gets a business-driven
decision threshold chosen on out-of-fold training predictions.

Outputs: models/churn_pipeline.joblib, models/churn_metadata.json, models/reference_profile.json,
         reports/model_results.md, reports/model_comparison.csv, reports/figures/model_*.png
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import ConfusionMatrixDisplay, average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, cross_val_score

from churnlib import config as C
from churnlib.features import FeatureBuilder
from churnlib.modeling import best_threshold, make_pipeline, metrics_at, realized_net_benefit

train = pd.read_csv(C.TRAIN_FILE); test = pd.read_csv(C.TEST_FILE)
cfg = json.loads((C.MODELS / "feature_config.json").read_text())
NUM, CAT, IMB, PW = cfg["numeric"], cfg["categorical"], cfg["imbalance_choice"], cfg["pos_weight"]
y_tr, y_te = train[C.TARGET].to_numpy(), test[C.TARGET].to_numpy()
cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)

SPACES = {
    "logreg": {"clf__C": [0.01, 0.1, 1, 10]},
    "rf": {"clf__n_estimators": [300, 500], "clf__max_depth": [6, 8, 12, None],
           "clf__min_samples_leaf": [1, 3, 5, 10], "clf__max_features": ["sqrt", 0.5]},
    "xgb": {"clf__n_estimators": [200, 300, 400, 600], "clf__max_depth": [3, 4, 5, 6],
            "clf__learning_rate": [0.02, 0.03, 0.05, 0.08], "clf__subsample": [0.7, 0.8, 0.9, 1.0],
            "clf__colsample_bytree": [0.6, 0.8, 1.0], "clf__min_child_weight": [1, 3, 5], "clf__reg_lambda": [1, 3, 10]},
}
N_ITER = {"logreg": 4, "rf": 8, "xgb": 14}

# ------------------------------------------------------------------ 1. tune + compare
results, fitted, oof = [], {}, {}
for name in ["logreg", "rf", "xgb"]:
    pipe = make_pipeline(name, NUM, CAT, IMB[name], PW)
    rs = RandomizedSearchCV(pipe, SPACES[name], n_iter=N_ITER[name], scoring="roc_auc", cv=cv,
                            random_state=C.SEED, n_jobs=1, refit=True)
    rs.fit(train, y_tr)
    fitted[name] = rs.best_estimator_
    p_te = rs.best_estimator_.predict_proba(test)[:, 1]
    m = metrics_at(y_te, p_te, 0.5)
    results.append({"model": name, "cv_roc_auc": rs.best_score_, **{f"test_{k}": v for k, v in m.items() if k != "threshold"},
                    "best_params": {k.replace("clf__", ""): v for k, v in rs.best_params_.items()}})
    print(name, "CV AUC", round(rs.best_score_, 4), "| test AUC", round(m["roc_auc"], 4), rs.best_params_)
res = pd.DataFrame(results)
res.drop(columns="best_params").round(4).to_csv(C.REPORTS / "model_comparison.csv", index=False)

best_name = res.sort_values("cv_roc_auc", ascending=False).iloc[0]["model"]
final = fitted[best_name]
print("FINAL MODEL:", best_name)

# ------------------------------------------------------------------ 2. does the synthetic layer make it trivial?
PROFILE_NUM = ["credit_score", "age", "tenure", "balance", "products_number", "credit_card", "active_member",
               "estimated_salary", "balance_to_salary", "products_per_tenure", "zero_balance_flag"]
profile_pipe = make_pipeline("xgb", PROFILE_NUM, ["country", "gender"], "none", PW)
xgb_params = res.set_index("model").loc["xgb", "best_params"]
profile_pipe.set_params(**{f"clf__{k}": v for k, v in xgb_params.items()})
profile_pipe.fit(train, y_tr)
auc_profile = roc_auc_score(y_te, profile_pipe.predict_proba(test)[:, 1])
auc_full = roc_auc_score(y_te, final.predict_proba(test)[:, 1])
print("profile-only AUC", round(auc_profile, 4), "| profile + behaviour AUC", round(auc_full, 4))

# ------------------------------------------------------------------ 3. calibration check
p_oof = cross_val_predict(final, train, y_tr, cv=cv, method="predict_proba")[:, 1]
p_te = final.predict_proba(test)[:, 1]
brier_oof = float(np.mean((p_oof - y_tr) ** 2))

# ------------------------------------------------------------------ 4. business threshold (on OOF train predictions)
clv_tr = C.clv_proxy(train.balance.to_numpy(), train.products_number.to_numpy())
clv_te = C.clv_proxy(test.balance.to_numpy(), test.products_number.to_numpy())
thr, curve = best_threshold(y_tr, p_oof, clv_tr)
print("business threshold:", round(thr, 3))

strategies = {
    "Contact nobody": np.zeros(len(test), bool),
    "Contact everybody": np.ones(len(test), bool),
    "Model, threshold 0.5": p_te >= 0.5,
    f"Model, business threshold {thr:.2f}": p_te >= thr,
}
strat_rows = []
for k, flag in strategies.items():
    tp = int(((y_te == 1) & flag).sum())
    strat_rows.append({"strategy": k, "customers_contacted": int(flag.sum()), "churners_reached": tp,
                       "recall": tp / (y_te == 1).sum(), "precision": tp / max(flag.sum(), 1),
                       "net_benefit_eur": realized_net_benefit(y_te, flag, clv_te)})
strat = pd.DataFrame(strat_rows)
print(strat.round(3).to_string(index=False))
m_final = metrics_at(y_te, p_te, thr)
m_final_05 = metrics_at(y_te, p_te, 0.5)

# ------------------------------------------------------------------ 5. figures
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
cols = {"logreg": "#7F7F7F", "rf": "#4C78A8", "xgb": "#E4572E"}
for name, est in fitted.items():
    p = est.predict_proba(test)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, p); axes[0].plot(fpr, tpr, color=cols[name], label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p); axes[1].plot(rc, pr, color=cols[name], label=f"{name} (AP {average_precision_score(y_te, p):.3f})")
axes[0].plot([0, 1], [0, 1], "k--", lw=.8); axes[0].set_title("ROC curves (test set)"); axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate"); axes[0].legend()
axes[1].set_title("Precision-recall curves (test set)"); axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision"); axes[1].legend()
fr, mp = calibration_curve(y_te, p_te, n_bins=10, strategy="quantile")
axes[2].plot(mp, fr, "o-", color="#E4572E", label=f"{best_name}"); axes[2].plot([0, 1], [0, 1], "k--", lw=.8)
axes[2].set_title("Calibration (test set)"); axes[2].set_xlabel("predicted probability"); axes[2].set_ylabel("observed churn rate"); axes[2].legend()
fig.tight_layout(); fig.savefig(C.FIGS / "model_curves.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ConfusionMatrixDisplay.from_predictions(y_te, (p_te >= thr).astype(int), display_labels=["stay", "churn"], cmap="Blues", ax=axes[0], colorbar=False)
axes[0].set_title(f"Confusion matrix at business threshold {thr:.2f}")
axes[1].plot(curve.threshold, curve.net_benefit / 1000, color="#4C78A8")
axes[1].axvline(thr, color="#E4572E", ls="--", label=f"chosen threshold {thr:.2f}")
axes[1].set_title("Net benefit of contacting customers (train, out-of-fold)"); axes[1].set_xlabel("decision threshold"); axes[1].set_ylabel("net benefit (thousand EUR)"); axes[1].legend()
fig.tight_layout(); fig.savefig(C.FIGS / "model_threshold.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 6. save artefacts
joblib.dump(final, C.MODELS / "churn_pipeline.joblib")
ref = {}
for c in C.PROFILE_COLS + C.BEHAVIOUR_COLS:
    if c in train and train[c].dtype != object and str(train[c].dtype) != "str":
        ref[c] = float(train[c].median())
    else:
        ref[c] = str(train[c].mode().iloc[0])
ref["login_missing_flag"] = 0
(C.MODELS / "reference_profile.json").write_text(json.dumps(ref, indent=2))
meta = {
    "model_version": C.MODEL_VERSION, "trained_at": datetime.now().isoformat(timespec="seconds"), "algorithm": best_name,
    "imbalance_strategy": IMB[best_name], "features_numeric": NUM, "features_categorical": CAT,
    "threshold": thr, "band_medium_from": round(thr, 3), "band_high_from": 0.5, "business": C.BUSINESS,
    "metrics_test_at_business_threshold": m_final, "metrics_test_at_0_5": m_final_05,
    "cv_roc_auc": float(res.set_index("model").loc[best_name, "cv_roc_auc"]),
    "auc_profile_only": auc_profile, "auc_profile_plus_behaviour": auc_full, "brier_oof": brier_oof,
    "train_rows": len(train), "test_rows": len(test), "train_churn_rate": float(y_tr.mean()),
}
(C.MODELS / "churn_metadata.json").write_text(json.dumps(meta, indent=2, default=float))

md = ["# Churn classifier - results", "", f"Final model: **{best_name}** (imbalance strategy: {IMB[best_name]}), model version {C.MODEL_VERSION}", "",
      "## Model comparison (tuned with 5-fold CV on train; test metrics at threshold 0.5)", "",
      res.drop(columns="best_params").round(4).to_markdown(index=False), "",
      "## Best hyper-parameters", ""] + [f"- {r.model}: `{r.best_params}`" for r in res.itertuples()] + \
     ["", f"## Test-set performance of the final model", "",
      pd.DataFrame([{"threshold": 0.5, **{k: round(v, 4) for k, v in m_final_05.items() if k != 'threshold'}},
                    {"threshold": round(thr, 3), **{k: round(v, 4) for k, v in m_final.items() if k != 'threshold'}}]).to_markdown(index=False), "",
      "## Business decision threshold", "",
      f"Chosen on out-of-fold training predictions by maximising realised net benefit: **{thr:.2f}**. Assumptions: {C.BUSINESS}.", "",
      strat.round(3).to_markdown(index=False), "",
      "## Does the synthetic behavioural layer make the problem trivial?", "",
      f"- test ROC-AUC with profile columns only: **{auc_profile:.3f}**",
      f"- test ROC-AUC with profile + behaviour: **{auc_full:.3f}**",
      f"- improvement: {auc_full - auc_profile:+.3f}. The behavioural layer helps but does not make the problem trivial; because the layer was "
      "generated with a churn-linked decline, part of this gain reflects the generator's assumptions and must not be read as real-world evidence.", "",
      f"## Calibration", "", f"Out-of-fold Brier score: {brier_oof:.4f} (always predicting the base rate gives {y_tr.mean() * (1 - y_tr.mean()):.4f})."]
(C.REPORTS / "model_results.md").write_text("\n".join(md), encoding="utf-8")
print("saved")
