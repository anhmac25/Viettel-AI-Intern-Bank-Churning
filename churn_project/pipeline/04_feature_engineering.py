"""Module 4 - Feature engineering.

1. stratified 80/20 split (before any fitting, to avoid leakage)
2. engineered features (FeatureBuilder, fitted on train only)
3. feature selection: drop one of every pair with |r| > 0.85, rank by mutual information
4. class-imbalance experiment: none vs class weights vs SMOTE (SMOTE only inside the CV training folds)

Outputs: data/processed/{train,test,features_all}.csv, models/feature_config.json,
         reports/{feature_ranking,imbalance_comparison}.csv, reports/figures/feature_*.png
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
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from churnlib import config as C
from churnlib.features import FeatureBuilder
from churnlib.modeling import make_pipeline

df = pd.read_csv(C.MERGED_CLEAN)
train, test = train_test_split(df, test_size=0.2, stratify=df[C.TARGET], random_state=C.SEED)
train.to_csv(C.TRAIN_FILE, index=False); test.to_csv(C.TEST_FILE, index=False)
print("train", train.shape, round(train.churn.mean(), 4), "| test", test.shape, round(test.churn.mean(), 4))

fb = FeatureBuilder().fit(train)
F = fb.transform(train)
F_all = pd.concat([df[[C.ID_COL]].reset_index(drop=True), fb.transform(df), df[[C.TARGET]].reset_index(drop=True)], axis=1)
F_all.to_csv(C.FEATURES_FILE, index=False)
y = train[C.TARGET].to_numpy()

# ------------------------------------------------------------------ correlation-based selection
num = FeatureBuilder.NUMERIC
corr = F[num].corr()
target_corr = F[num].apply(lambda s: abs(np.corrcoef(s, y)[0, 1]))
PROTECTED = {"balance", "age", "active_member", "products_number"}
dropped, keep = [], list(num)
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(num) for b in num[i + 1:] if abs(corr.loc[a, b]) > 0.85]
for a, b, r in sorted(pairs, key=lambda t: -abs(t[2])):
    if a in keep and b in keep:
        loser = a if target_corr[a] < target_corr[b] else b
        if loser in PROTECTED:                      # never drop core business variables
            loser = b if loser == a else a
        winner = b if loser == a else a
        keep.remove(loser)
        dropped.append({"dropped": loser, "kept": winner, "abs_corr": round(abs(r), 3)})
print("dropped (|r|>0.85):", dropped)

fig, ax = plt.subplots(figsize=(13, 11))
sns.heatmap(corr, cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": .7}, xticklabels=True, yticklabels=True)
ax.tick_params(labelsize=7); ax.set_title("Correlation of engineered numeric features (train set)")
fig.tight_layout(); fig.savefig(C.FIGS / "feature_correlation.png", dpi=120); plt.close(fig)

# ------------------------------------------------------------------ mutual information ranking
Xmi = pd.concat([F[keep], pd.get_dummies(F[FeatureBuilder.CATEGORICAL], dtype=float)], axis=1)
mi = pd.Series(mutual_info_classif(Xmi, y, random_state=C.SEED), index=Xmi.columns).sort_values(ascending=False)
mi.rename("mutual_information").to_csv(C.REPORTS / "feature_ranking.csv")
fig, ax = plt.subplots(figsize=(8, 8))
mi.head(20)[::-1].plot.barh(ax=ax, color="#4C78A8")
ax.set_title("Top 20 features by mutual information with churn (train)"); ax.set_xlabel("mutual information")
fig.tight_layout(); fig.savefig(C.FIGS / "feature_mutual_information.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ class imbalance experiment
pos_weight = float((y == 0).sum() / (y == 1).sum())
cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "recall": "recall", "precision": "precision", "brier": "neg_brier_score"}
rows = []
for model in ["logreg", "rf", "xgb"]:
    for imb in ["none", "class_weight", "smote"]:
        pipe = make_pipeline(model, keep, FeatureBuilder.CATEGORICAL, imb, pos_weight)
        r = cross_validate(pipe, train, train[C.TARGET], cv=cv, scoring=scoring, n_jobs=1)
        rows.append({"model": model, "imbalance": imb, **{k: round(float(np.mean(r[f"test_{k}"])) * (-1 if k == "brier" else 1), 4) for k in scoring}})
        print(rows[-1])
res = pd.DataFrame(rows)
res.to_csv(C.REPORTS / "imbalance_comparison.csv", index=False)

# choose per model: best ROC-AUC; if within 0.003 of 'none', keep 'none' (better calibrated probabilities)
choice = {}
for m, g in res.groupby("model"):
    g = g.set_index("imbalance")
    best = g.roc_auc.idxmax()
    choice[m] = "none" if g.loc["none", "roc_auc"] >= g.loc[best, "roc_auc"] - 0.003 else best
print("imbalance choice:", choice)

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, metric in zip(axes, ["roc_auc", "recall", "brier"]):
    sns.barplot(data=res, x="model", y=metric, hue="imbalance", ax=ax)
    ax.set_title({"roc_auc": "CV ROC-AUC", "recall": "CV recall @0.5", "brier": "CV Brier (lower = better calibrated)"}[metric])
    if metric == "roc_auc": ax.set_ylim(0.7, 1.0)
fig.tight_layout(); fig.savefig(C.FIGS / "feature_imbalance_comparison.png", dpi=130); plt.close(fig)

cfg = {"numeric": keep, "categorical": FeatureBuilder.CATEGORICAL, "dropped_correlated": dropped,
       "imbalance_choice": choice, "pos_weight": pos_weight}
(C.MODELS / "feature_config.json").write_text(json.dumps(cfg, indent=2))
print("selected numeric features:", len(keep))
