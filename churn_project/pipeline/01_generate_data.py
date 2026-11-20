"""Module 1 - Business understanding & data generation.

Outputs
  data/interim/behaviour_raw.csv          raw (dirty) synthetic behaviour export
  docs/data_dictionary.md / .csv          data dictionary
  reports/synthetic_validation.md         realism checks
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from churnlib import config as C
from churnlib.dictionary import dictionary_df, to_markdown
from churnlib.synth import generate_behaviour

base = pd.read_csv(C.BASE_FILE)
print("base:", base.shape, "| churn rate:", round(base.churn.mean(), 4))

clean_ref = generate_behaviour(base, add_noise=False)     # defect-free copy only for validation
raw = generate_behaviour(base, add_noise=True)
raw.to_csv(C.BEHAVIOUR_RAW, index=False)
print("raw behaviour table:", raw.shape)

(C.DOCS / "data_dictionary.md").write_text(to_markdown(), encoding="utf-8")
dictionary_df().to_csv(C.DOCS / "data_dictionary.csv", index=False)

# ------------------------------------------------------------------ realism checks
m = base.merge(clean_ref, on="customer_id")
cnt, amt = C.TXN_COUNT_COLS, C.TXN_AMOUNT_COLS
checks = []
checks.append(("No negative counts", bool((m[cnt] >= 0).all().all())))
checks.append(("No negative amounts (defect-free copy)", bool((m[amt] >= 0).all().all())))
checks.append(("Channel shares sum to 1 (+/-0.001)", bool(np.allclose(m[["atm_share", "mobile_share", "branch_share"]].sum(axis=1), 1, atol=1e-3))))
checks.append(("0 logins => >= 30 days since last login", bool((m.loc[m.login_count_30d == 0, "days_since_last_login"] >= 30).all())))
checks.append(("0 transactions in m6 => >= 30 days since last txn", bool((m.loc[m.txn_count_m6 == 0, "days_since_last_txn"] >= 30).all())))

m["freq6"] = m[cnt].sum(axis=1)
m["ratio"] = (m[cnt[3:]].mean(axis=1) + 1) / (m[cnt[:3]].mean(axis=1) + 1)
corr = {
    "corr(txn frequency, logins)": m["freq6"].corr(m["login_count_30d"]),
    "corr(complaints, churn)": m["complaints_6m"].corr(m["churn"]),
    "corr(recent/early txn ratio, churn)": m["ratio"].corr(m["churn"]),
    "corr(logins, churn)": m["login_count_30d"].corr(m["churn"]),
    "corr(age, mobile share)": m["age"].corr(m["mobile_share"]),
}
signs_ok = (corr["corr(txn frequency, logins)"] > 0 and corr["corr(complaints, churn)"] > 0
            and corr["corr(recent/early txn ratio, churn)"] < 0 and corr["corr(logins, churn)"] < 0
            and corr["corr(age, mobile share)"] < 0)
checks.append(("Correlation signs match business expectation", bool(signs_ok)))

lines = ["# Synthetic data validation", "", "## Automated checks (defect-free copy)", ""]
lines += [f"- {'PASS' if ok else 'FAIL'} - {name}" for name, ok in checks]
lines += ["", "## Correlations", ""] + [f"- {k}: {v:+.3f}" for k, v in corr.items()]
g = m.groupby("churn")[["freq6", "ratio", "login_count_30d", "complaints_6m", "days_since_last_txn"]].mean().round(3)
lines += ["", "## Group means (0 = retained, 1 = churned)", "", g.to_markdown(), "",
          "Churners and retained customers overlap by design (30% silent churners, 10% temporary dips); "
          "see reports/model_results.md for the profile-only vs combined model comparison that tests whether the synthetic layer makes the problem trivial.",
          "", "## Defects injected in the raw table", "",
          f"- rows raw: {len(raw)} vs customers: {len(base)}",
          f"- missing login_count_30d: {int(raw.login_count_30d.isna().sum())}",
          f"- exact duplicate rows: {int(raw.duplicated().sum())}",
          f"- distinct channel labels: {raw.preferred_channel.nunique()}",
          f"- negative monthly amounts: {int((raw[amt] < 0).sum().sum())}",
          f"- ids not in customer base: {int((~raw.customer_id.isin(base.customer_id)).sum())}"]
(C.REPORTS / "synthetic_validation.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines[:12]))
assert all(ok for _, ok in checks), "validation failed"
