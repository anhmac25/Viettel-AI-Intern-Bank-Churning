"""Module 5 / Branch B - customer segmentation (K-Means on RFM / behavioural features) + integration.

Outputs: models/segmenter.joblib, models/personas.json, data/processed/customers_segmented.csv,
         reports/segmentation_results.md, reports/figures/segment_*.png
"""
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from churnlib import config as C
from churnlib.modeling import make_preprocessor
from churnlib.segmentation import SEG_FEATURES, Segmenter, _matrix, assign_personas, describe

K_CHOSEN = 5
train = pd.read_csv(C.TRAIN_FILE); test = pd.read_csv(C.TEST_FILE); allc = pd.read_csv(C.MERGED_CLEAN)
final = joblib.load(C.MODELS / "churn_pipeline.joblib")
fb = final.named_steps["features"]                      # same fitted FeatureBuilder as the classifier
F_tr, F_te, F_all = fb.transform(train), fb.transform(test), fb.transform(allc)

# ------------------------------------------------------------------ 1. choosing k
X = StandardScaler().fit_transform(_matrix(F_tr))
ks, inertia, sil = list(range(2, 9)), [], []
for k in ks:
    km = KMeans(k, n_init=10, random_state=C.SEED).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_, sample_size=4000, random_state=1))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(ks, inertia, "o-", color="#4C78A8"); axes[0].set_title("Elbow: within-cluster sum of squares"); axes[0].set_xlabel("k")
axes[1].plot(ks, sil, "o-", color="#E4572E"); axes[1].axvline(K_CHOSEN, ls="--", color="grey"); axes[1].set_title("Silhouette score"); axes[1].set_xlabel("k")
fig.tight_layout(); fig.savefig(C.FIGS / "segment_k_selection.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 2. fit + personas
seg = Segmenter(K_CHOSEN, C.SEED).fit(F_tr)
lab_tr, lab_te, lab_all = seg.predict(F_tr), seg.predict(F_te), seg.predict(F_all)
z = seg.zscores(F_tr, lab_tr)

TEMPLATES = {   # target z-scores that define each persona (only listed features are compared)
    "Digital Power Users":        {"txn_freq_6m": 1.3, "login_count_30d": 1.0, "txn_amount_6m": 1.1},
    "Branch-Oriented Seniors":    {"age": 1.4, "mobile_share": -1.1, "login_count_30d": -0.5},
    "Dormant & Fading":           {"recency_days": 3.0, "txn_ratio_recent": -1.0, "txn_freq_6m": -1.0},
    "High-Balance Savers":        {"balance": 0.7, "products_number": -0.5, "age": -0.3},
    "Everyday Low-Balance Users": {"balance": -1.1, "products_number": 0.6},
}
cluster_to_persona = assign_personas(z, TEMPLATES)
ACTIONS = {
    "Digital Power Users": {
        "low": "No contact. Keep monitoring; cross-sell through in-app offers.",
        "medium": "In-app nudge or cashback on their favourite spending category; avoid costly discounts.",
        "high": "Alert the digital-engagement team: personalised in-app offer and a short survey about what changed."},
    "Branch-Oriented Seniors": {
        "low": "No contact. Offer help to use digital services at the next branch visit.",
        "medium": "Friendly call or letter from the branch; guided digital onboarding.",
        "high": "Personal call from the branch relationship manager within one week; priority service line and fee review."},
    "Dormant & Fading": {
        "low": "Light-touch reactivation email.",
        "medium": "Reactivation campaign with a simple first action (e.g. set up a direct debit) and a small incentive.",
        "high": "Immediate outbound contact: understand the reason, resolve open complaints, offer a targeted win-back incentive."},
    "High-Balance Savers": {
        "low": "No contact. Annual relationship review.",
        "medium": "Relationship-manager check-in; competitive savings or investment offer.",
        "high": "Priority call by a senior relationship manager; bespoke retention offer protecting the balance."},
    "Everyday Low-Balance Users": {
        "low": "No contact; promote savings goals in the app.",
        "medium": "Low-cost automated offer: fee waiver or salary-deposit bonus.",
        "high": "Review products and fees (many products, low balance); proactive complaint resolution; simplify the product bundle."},
}

prof = seg.profile(F_tr, lab_tr)
info = {}
for cl, persona in cluster_to_persona.items():
    m = lab_tr == cl
    info[persona] = {
        "cluster": int(cl), "description": describe(z.loc[cl]),
        "share_of_customers": float((lab_all == cl).mean()),
        "churn_rate_train": float(train.churn[m].mean()), "churn_rate_test": float(test.churn[lab_te == cl].mean()),
        "churn_rate_all": float(allc.churn[lab_all == cl].mean()),
        "profile": {k: float(v) for k, v in prof.loc[cl].items()}, "actions": ACTIONS[persona],
    }
joblib.dump({"segmenter": seg, "cluster_to_persona": cluster_to_persona, "k": K_CHOSEN}, C.MODELS / "segmenter.joblib")
(C.MODELS / "personas.json").write_text(json.dumps(info, indent=2))

seg_df = pd.DataFrame({C.ID_COL: allc[C.ID_COL], "segment": lab_all, "persona": [cluster_to_persona[c] for c in lab_all], C.TARGET: allc[C.TARGET]})
seg_df.to_csv(C.DATA_PROC / "customers_segmented.csv", index=False)

# ------------------------------------------------------------------ 3. figures
order = sorted(info, key=lambda p: -info[p]["churn_rate_all"])
zz = z.rename(index=cluster_to_persona).loc[order]
fig, ax = plt.subplots(figsize=(10, 4.2))
sns.heatmap(zz, annot=True, fmt=".1f", cmap="RdBu_r", center=0, ax=ax, cbar_kws={"label": "z-score vs all customers"})
ax.set_title("Segment profiles (standardised cluster means)"); ax.set_ylabel(""); plt.xticks(rotation=30, ha="right")
fig.tight_layout(); fig.savefig(C.FIGS / "segment_profiles.png", dpi=130); plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
cr = pd.Series({p: info[p]["churn_rate_all"] * 100 for p in order}); sh = pd.Series({p: info[p]["share_of_customers"] * 100 for p in order})
sns.barplot(x=cr.values, y=cr.index, ax=axes[0], color="#E4572E"); axes[0].set_title("Churn rate by segment (%)"); axes[0].axvline(allc.churn.mean() * 100, ls="--", color="k", lw=.8)
for i, v in enumerate(cr.values): axes[0].text(v + 0.5, i, f"{v:.0f}%", va="center")
sns.barplot(x=sh.values, y=sh.index, ax=axes[1], color="#4C78A8"); axes[1].set_title("Share of customers (%)"); axes[1].set_ylabel("")
for i, v in enumerate(sh.values): axes[1].text(v + 0.3, i, f"{v:.0f}%", va="center")
fig.tight_layout(); fig.savefig(C.FIGS / "segment_churn_share.png", dpi=130); plt.close(fig)

pts = PCA(2, random_state=C.SEED).fit_transform(StandardScaler().fit_transform(_matrix(F_tr)))
fig, ax = plt.subplots(figsize=(7.5, 5.5))
for cl, persona in cluster_to_persona.items():
    ax.scatter(*pts[lab_tr == cl].T, s=6, alpha=.5, label=persona)
ax.legend(markerscale=3, fontsize=8); ax.set_title("Segments in 2-D (PCA projection)"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
fig.tight_layout(); fig.savefig(C.FIGS / "segment_pca.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 4. does the segment help the classifier?
meta = json.loads((C.MODELS / "churn_metadata.json").read_text())
cfg = json.loads((C.MODELS / "feature_config.json").read_text())
prep = make_preprocessor(cfg["numeric"], cfg["categorical"]).fit(F_tr)
A = prep.transform(F_tr)
S = pd.get_dummies(pd.Series(lab_tr).astype(str), dtype=float).to_numpy()
clf_params = {k.replace("clf__", ""): v for k, v in final.named_steps["clf"].get_params().items()
              if k in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "min_child_weight", "reg_lambda")}
cv = StratifiedKFold(5, shuffle=True, random_state=C.SEED)
mk = lambda: XGBClassifier(eval_metric="logloss", random_state=C.SEED, n_jobs=1, **clf_params)
y = train[C.TARGET].to_numpy()
auc_without = cross_val_score(mk(), A, y, cv=cv, scoring="roc_auc").mean()
auc_with = cross_val_score(mk(), np.hstack([A, S]), y, cv=cv, scoring="roc_auc").mean()
print(f"CV AUC without segment {auc_without:.4f} | with segment {auc_with:.4f} | delta {auc_with - auc_without:+.4f}")

# ------------------------------------------------------------------ 5. report
tab = pd.DataFrame({
    "persona": order, "cluster": [info[p]["cluster"] for p in order],
    "share %": [round(info[p]["share_of_customers"] * 100, 1) for p in order],
    "churn % (all)": [round(info[p]["churn_rate_all"] * 100, 1) for p in order],
    "churn % (train)": [round(info[p]["churn_rate_train"] * 100, 1) for p in order],
    "churn % (test)": [round(info[p]["churn_rate_test"] * 100, 1) for p in order],
    "profile (vs average)": [info[p]["description"] for p in order]})
profile_means = pd.DataFrame({p: info[p]["profile"] for p in order}).T.round(2)
md = ["# Segmentation results", "", f"K-Means with **k = {K_CHOSEN}** on {len(SEG_FEATURES)} standardised behavioural/RFM features "
      "(log transform on spending and balance); fitted on the training set; the churn label is NOT used.", "",
      "## Choosing k", "", pd.DataFrame({"k": ks, "inertia": np.round(inertia), "silhouette": np.round(sil, 3)}).to_markdown(index=False), "",
      "Silhouette scores are low for every k (customer behaviour is a continuum, without sharp natural groups), so k was chosen for "
      f"interpretability: k = {K_CHOSEN} has the best (or tied-best) silhouette among k >= 4 and is the first solution that isolates a small, "
      "high-risk 'dormant & fading' group.", "", "## Personas", "", tab.to_markdown(index=False), "",
      "## Average profile per persona (original units)", "", profile_means.to_markdown(), "",
      "## Segment as a classifier feature", "",
      f"5-fold CV ROC-AUC without segment: **{auc_without:.4f}**; with segment: **{auc_with:.4f}** (delta {auc_with - auc_without:+.4f}). "
      "The segment adds little predictive power, because it is built from the same behaviour the classifier already sees - its value is "
      "interpretability and targeting, so the deployed classifier does not use it.", "",
      "## Retention actions by persona and risk band", ""]
for p in order:
    md += [f"### {p}", ""] + [f"- **{b}:** {a}" for b, a in ACTIONS[p].items()] + [""]
(C.REPORTS / "segmentation_results.md").write_text("\n".join(md), encoding="utf-8")
print(tab.to_string(index=False))
