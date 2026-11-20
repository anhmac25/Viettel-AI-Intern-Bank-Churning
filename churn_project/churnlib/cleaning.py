"""Data cleaning (Module 3). The row-level rules are reused at serving time so that a customer
scored in the app is treated exactly like a customer in the training data."""
import json

import numpy as np
import pandas as pd

from .config import PROFILE_COLS, TXN_AMOUNT_COLS, TXN_COUNT_COLS

CANON = {"atm": "ATM", "mobile": "Mobile app", "mobile app": "Mobile app", "branch": "Branch"}
VALID = {"credit_score": (300, 850), "age": (18, 100), "tenure": (0, 10),
         "products_number": (1, 4), "balance": (0, np.inf), "estimated_salary": (0, np.inf)}


def canon_channel(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().map(CANON)


def fit_params(df: pd.DataFrame) -> dict:
    """Parameters learnt from (already de-duplicated) data and stored for serving."""
    amt = df[TXN_AMOUNT_COLS].where(df[TXN_AMOUNT_COLS] > 0).to_numpy().ravel()
    amt = amt[~np.isnan(amt)]
    login_med = df.groupby("active_member")["login_count_30d"].median().to_dict()
    dlogin_med = df.groupby("active_member")["days_since_last_login"].median().to_dict()
    return {
        "amount_cap": float(np.percentile(amt, 99.5)),
        "login_median_by_active": {str(int(k)): float(v) for k, v in login_med.items()},
        "days_login_median_by_active": {str(int(k)): float(v) for k, v in dlogin_med.items()},
    }


def apply_row_rules(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Row-level rules: channel labels, negative amounts, extreme amounts, missing logins."""
    df = df.copy()
    df["preferred_channel"] = canon_channel(df["preferred_channel"])

    amt = df[TXN_AMOUNT_COLS].astype(float)
    amt = amt.mask(amt < 0)                                   # reversals -> missing
    row_med = amt.where(amt > 0).median(axis=1).fillna(0.0)   # customer's own typical month
    for c in TXN_AMOUNT_COLS:
        amt[c] = amt[c].fillna(row_med)
    df[TXN_AMOUNT_COLS] = amt.clip(upper=params["amount_cap"])  # winsorise extreme values

    flag = df["login_count_30d"].isna().astype(int)
    if "login_missing_flag" in df:                      # keep a flag set earlier (e.g. customer already cleaned)
        flag = np.maximum(flag, df["login_missing_flag"].astype(int))
    df["login_missing_flag"] = flag
    act = df["active_member"].astype(int).astype(str)
    df["login_count_30d"] = df["login_count_30d"].fillna(act.map(params["login_median_by_active"]))
    df["days_since_last_login"] = df["days_since_last_login"].fillna(act.map(params["days_login_median_by_active"]))
    return df


def merge_and_clean(base: pd.DataFrame, raw_beh: pd.DataFrame):
    """Returns (clean_df, log_df, params). Each log row: issue, check, treatment, before, after."""
    log = []

    def add(issue, check, treatment, before, after):
        log.append({"issue": issue, "how_detected": check, "treatment": treatment, "before": before, "after": after})

    # ---- base (Kaggle) audit ---------------------------------------------------------
    add("Missing values (Kaggle)", "isna().sum()", "none needed", int(base.isna().sum().sum()), int(base.isna().sum().sum()))
    add("Duplicate customers (Kaggle)", "duplicated(customer_id)", "none needed", int(base.customer_id.duplicated().sum()),
        int(base.customer_id.duplicated().sum()))
    bad = np.zeros(len(base), bool)
    for c, (lo, hi) in VALID.items():
        bad |= ~base[c].between(lo, hi)
    add("Invalid ages / balances / scores (Kaggle)", "range rules: age 18-100, balance >= 0, score 300-850, tenure 0-10, products 1-4",
        "drop invalid rows", int(bad.sum()), 0)
    base = base.loc[~bad].copy()
    add("Inconsistent categories (Kaggle)", "unique values of country, gender",
        "none needed", int(base.country.nunique() + base.gender.nunique()), int(base.country.nunique() + base.gender.nunique()))

    beh = raw_beh.copy()
    # ---- 1. exact duplicates ---------------------------------------------------------
    n0 = int(beh.duplicated().sum()); beh = beh.drop_duplicates()
    add("Duplicate behaviour rows", "duplicated() on all columns", "keep first occurrence", n0, int(beh.duplicated().sum()))
    n0 = int(beh.customer_id.duplicated().sum()); beh = beh.drop_duplicates("customer_id")
    add("Duplicate customer_id after de-duplication", "duplicated(customer_id)", "keep first", n0, int(beh.customer_id.duplicated().sum()))

    # ---- 2. accounts that are not customers -----------------------------------------
    orphan = ~beh.customer_id.isin(base.customer_id)
    add("Ids not in the customer base (test accounts)", "anti-join on customer_id", "drop", int(orphan.sum()), 0)
    beh = beh.loc[~orphan]

    # ---- 3. channel labels -----------------------------------------------------------
    before = int(beh.preferred_channel.nunique())
    after_lbl = canon_channel(beh.preferred_channel).nunique()
    add("Inconsistent channel labels", "unique values of preferred_channel", "strip + lower-case + map to 3 canonical labels", before, int(after_lbl))

    # ---- merge (needed because the login imputation uses the activity flag) ----------
    df = base.merge(beh, on="customer_id", how="inner", validate="one_to_one")
    add("Customers without behavioural history after merge", "inner join size vs base", "none (all customers matched)",
        int(len(base) - len(df)), int(len(base) - len(df)))

    # ---- 4-6. counted before the row rules are applied ---------------------------------
    n_neg = int((df[TXN_AMOUNT_COLS] < 0).sum().sum())
    params = fit_params(df)
    n_ext = int((df[TXN_AMOUNT_COLS] > params["amount_cap"]).sum().sum())
    n_miss = int(df.login_count_30d.isna().sum())

    df = apply_row_rules(df, params)
    add("Negative monthly amounts (reversal artefacts)", "amount < 0", "set to missing, impute with the customer's own median month", n_neg,
        int((df[TXN_AMOUNT_COLS] < 0).sum().sum()))
    add("Extreme monthly amounts (x20 entry errors)", "amount > 99.5th percentile of positive amounts "
        f"({params['amount_cap']:.0f} EUR)", "winsorise at the 99.5th percentile; original values not kept in modelling file", n_ext,
        int((df[TXN_AMOUNT_COLS] > params['amount_cap']).sum().sum()))
    add("Missing login data (no app tracking)", "isna() on login_count_30d", "impute median by activity status + login_missing_flag",
        n_miss, int(df.login_count_30d.isna().sum()))

    # ---- final consistency checks --------------------------------------------------------
    assert df[TXN_COUNT_COLS].ge(0).all().all()
    assert df[TXN_AMOUNT_COLS].ge(0).all().all()
    assert not df.drop(columns=[]).isna().any().any()
    assert df.customer_id.is_unique
    return df.reset_index(drop=True), pd.DataFrame(log), params


def save_params(params: dict, path):
    with open(path, "w") as f:
        json.dump(params, f, indent=2)
