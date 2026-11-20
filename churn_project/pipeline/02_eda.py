"""Module 2 - Exploratory Data Analysis (on the merged but NOT yet cleaned data).

Light preparation only (exact duplicates and non-customer ids are removed by the join) so that the
EDA can also expose the defects the cleaning step must treat.

Outputs: reports/figures/eda_*.png and reports/eda_summary.md
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from churnlib import config as C

sns.set_theme(style="whitegrid", font_scale=0.95)
PAL = {0: "#4C78A8", 1: "#E4572E"}

base = pd.read_csv(C.BASE_FILE)
raw = pd.read_csv(C.BEHAVIOUR_RAW).drop_duplicates("customer_id")
df = base.merge(raw, on="customer_id", how="inner")
cnt, amt = C.TXN_COUNT_COLS, C.TXN_AMOUNT_COLS
df[amt] = df[amt].clip(lower=0)                      # only for plotting
df["txn_freq_6m"] = df[cnt].sum(axis=1)
df["txn_amount_6m"] = df[amt].sum(axis=1)
df["age_band"] = pd.cut(df.age, [17, 29, 39, 49, 59, 100], labels=["18-29", "30-39", "40-49", "50-59", "60+"])
df["balance_band"] = np.where(df.balance == 0, "zero", pd.qcut(df.balance.replace(0, np.nan), 4,
                              labels=["low", "mid-low", "mid-high", "high"]).astype(str))
df["balance_band"] = pd.Categorical(df["balance_band"], categories=["zero", "low", "mid-low", "mid-high", "high"], ordered=True)
df["tenure_band"] = pd.cut(df.tenure, [-1, 2, 5, 8, 10], labels=["0-2", "3-5", "6-8", "9-10"])
overall = df.churn.mean()
md = ["# EDA summary", "", f"Customers: {len(df):,} | overall churn rate: **{overall:.1%}**", ""]


def rate_table(col):
    t = df.groupby(col, observed=True).churn.agg(rate="mean", customers="count").round(4)
    t["rate"] = (t.rate * 100).round(1)
    return t


# ------------------------------------------------------------------ 1. churn by segment (grid)
dims = [("country", "Country"), ("gender", "Gender"), ("products_number", "Number of products"),
        ("active_member", "Active member"), ("age_band", "Age band"), ("tenure_band", "Tenure (years)"),
        ("balance_band", "Balance band"), ("credit_card", "Has credit card")]
fig, axes = plt.subplots(2, 4, figsize=(15, 7))
for ax, (col, title) in zip(axes.ravel(), dims):
    t = rate_table(col)
    sns.barplot(x=t.index.astype(str), y=t["rate"], ax=ax, color="#4C78A8")
    ax.axhline(overall * 100, color="#E4572E", ls="--", lw=1)
    ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("churn rate (%)")
    for i, v in enumerate(t["rate"]):
        ax.text(i, v + 0.6, f"{v:.0f}", ha="center", fontsize=8)
    md += [f"## Churn rate by {title.lower()}", "", t.to_markdown(), ""]
fig.suptitle(f"Churn rate by customer group (dashed line = overall {overall:.1%})", fontsize=13)
fig.tight_layout(); fig.savefig(C.FIGS / "eda_churn_by_group.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 2. distributions
fig, axes = plt.subplots(2, 3, figsize=(14, 7))
for ax, col in zip(axes.ravel(), ["age", "credit_score", "balance", "estimated_salary", "txn_freq_6m", "login_count_30d"]):
    for k, g in df.groupby("churn"):
        sns.kdeplot(g[col].dropna(), ax=ax, color=PAL[k], fill=True, alpha=0.25, label="churned" if k else "retained", cut=0)
    ax.set_title(col); ax.legend()
fig.tight_layout(); fig.savefig(C.FIGS / "eda_distributions.png", dpi=130); plt.close(fig)
zero_bal = (df.balance == 0).mean()
md += ["## Distribution notes", "",
       f"- {zero_bal:.1%} of customers hold a zero balance (a real dormant-account behaviour, kept as a feature).",
       f"- Age: median {df.age.median():.0f}, max {df.age.max()} (all within 18-100).",
       f"- Salary shows almost no difference between churners and retained customers "
       f"(median {df[df.churn==1].estimated_salary.median():,.0f} vs {df[df.churn==0].estimated_salary.median():,.0f}).", ""]

# ------------------------------------------------------------------ 3. correlation heatmap
num = df[["credit_score", "age", "tenure", "balance", "products_number", "active_member", "estimated_salary",
          "txn_freq_6m", "txn_amount_6m", "login_count_30d", "days_since_last_txn", "complaints_6m",
          "mobile_share", "churn"]]
fig, ax = plt.subplots(figsize=(9.5, 7.5))
sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax, annot_kws={"size": 7.5}, cbar_kws={"shrink": .8})
ax.set_title("Correlation matrix (numeric variables)")
fig.tight_layout(); fig.savefig(C.FIGS / "eda_correlation.png", dpi=130); plt.close(fig)
cc = num.corr()["churn"].drop("churn").sort_values(key=abs, ascending=False)
md += ["## Correlation with churn (Pearson)", "", cc.round(3).to_frame("corr").to_markdown(), ""]

# ------------------------------------------------------------------ 4. behavioural trend
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
months = np.arange(1, 7)
trend = df.groupby("churn")[cnt].mean()
for k in (0, 1):
    axes[0].plot(months, trend.loc[k].values, marker="o", color=PAL[k], label="churned" if k else "retained")
axes[0].set_title("Average monthly transactions (months 1 = oldest, 6 = latest)")
axes[0].set_xlabel("month"); axes[0].set_ylabel("transactions"); axes[0].legend()
trend_a = df.groupby("churn")[amt].median()
for k in (0, 1):
    axes[1].plot(months, trend_a.loc[k].values, marker="o", color=PAL[k], label="churned" if k else "retained")
axes[1].set_title("Median monthly amount (EUR)"); axes[1].set_xlabel("month"); axes[1].legend()
fig.tight_layout(); fig.savefig(C.FIGS / "eda_behaviour_trend.png", dpi=130); plt.close(fig)
md += ["## Behavioural trend (mean monthly transactions)", "", trend.round(2).to_markdown(), ""]

# ------------------------------------------------------------------ 5. RFM-style analysis
df["R"] = pd.qcut(-df.days_since_last_txn, 5, labels=False, duplicates="drop") + 1       # 5 = most recent
df["F"] = pd.qcut(df.txn_freq_6m.rank(method="first"), 5, labels=False) + 1
df["M"] = pd.qcut(df.txn_amount_6m.rank(method="first"), 5, labels=False) + 1
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, name in zip(axes, ["R", "F", "M"], ["Recency (5 = most recent)", "Frequency (5 = most transactions)", "Monetary (5 = highest amount)"]):
    t = df.groupby(col).churn.mean() * 100
    sns.barplot(x=t.index, y=t.values, ax=ax, color="#4C78A8")
    ax.axhline(overall * 100, color="#E4572E", ls="--", lw=1)
    ax.set_title(name); ax.set_ylabel("churn rate (%)"); ax.set_xlabel("quintile")
    md += [f"### Churn rate by {name}", "", t.round(1).to_frame("churn %").to_markdown(), ""]
fig.tight_layout(); fig.savefig(C.FIGS / "eda_rfm.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 6. logins & complaints
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
t = df.assign(lb=pd.cut(df.login_count_30d, [-1, 0, 5, 10, 20, 200], labels=["0", "1-5", "6-10", "11-20", "21+"])).groupby("lb", observed=True).churn.mean() * 100
sns.barplot(x=t.index.astype(str), y=t.values, ax=axes[0], color="#4C78A8"); axes[0].set_title("Churn rate by logins in 30 days"); axes[0].set_ylabel("%")
t2 = df.assign(cb=df.complaints_6m.clip(upper=3)).groupby("cb").churn.mean() * 100
sns.barplot(x=["0", "1", "2", "3+"], y=t2.values, ax=axes[1], color="#4C78A8"); axes[1].set_title("Churn rate by complaints (6 months)")
fig.tight_layout(); fig.savefig(C.FIGS / "eda_engagement.png", dpi=130); plt.close(fig)

# ------------------------------------------------------------------ 7. data-quality audit of the raw table
audit = [
    ("missing login_count_30d", int(df.login_count_30d.isna().sum())),
    ("negative monthly amounts (raw)", int((pd.read_csv(C.BEHAVIOUR_RAW)[amt] < 0).sum().sum())),
    ("amounts above 99.5th percentile (raw)", int((pd.read_csv(C.BEHAVIOUR_RAW)[amt] > pd.read_csv(C.BEHAVIOUR_RAW)[amt].stack().quantile(0.995)).sum().sum())),
    ("distinct channel labels", int(df.preferred_channel.nunique())),
]
md += ["## Data-quality findings that feed Module 3", ""] + [f"- {k}: {v}" for k, v in audit] + [""]

# ------------------------------------------------------------------ 8. key insights (computed, not typed)
c = rate_table("country")["rate"]; p = rate_table("products_number")["rate"]; a = rate_table("active_member")["rate"]
g = rate_table("gender")["rate"]; ab = rate_table("age_band")["rate"]
md += ["## Key insights", "",
       f"1. **Country:** {c.idxmax()} churns at {c.max():.1f}% vs {c.min():.1f}% in {c.idxmin()}.",
       f"2. **Products:** 1 product {p[1]:.1f}%, 2 products {p[2]:.1f}%, 3 products {p[3]:.1f}%, 4 products {p[4]:.1f}% - two products is the healthy level.",
       f"3. **Activity:** inactive members churn at {a[0]:.1f}% vs {a[1]:.1f}% for active members.",
       f"4. **Age:** the highest-risk band is {ab.idxmax()} ({ab.max():.1f}%); the lowest is {ab.idxmin()} ({ab.min():.1f}%).",
       f"5. **Gender:** female customers churn at {g['Female']:.1f}% vs {g['Male']:.1f}% for male customers.",
       f"6. **Behaviour:** churners start at a lower level ({trend.loc[1, cnt[0]]:.1f} vs {trend.loc[0, cnt[0]]:.1f} transactions in month 1, partly because they are less often "
       f"active members) and then fall by {(1 - trend.loc[1, cnt[-1]] / trend.loc[1, cnt[0]]):.0%} by month 6, while retained customers stay almost flat "
       f"({(1 - trend.loc[0, cnt[-1]] / trend.loc[0, cnt[0]]):.0%} change) - the TREND adds information beyond the level.",
       "7. **Salary and credit card ownership** show almost no relationship with churn."]
(C.REPORTS / "eda_summary.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[-12:]))
