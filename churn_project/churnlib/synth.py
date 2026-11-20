"""Synthetic behavioural data generator (Module 1).

The Kaggle file only describes WHO the customer is. A digital bank also sees HOW the customer
behaves, so we generate a six-month behavioural layer on top of the profile.

Design principles (all documented in docs/data_dictionary.md):
1. ANCHOR ON THE PROFILE - baseline activity depends on activity status, age, products, salary...
2. CHURN IS A PROCESS, NOT A SWITCH - most churners show a gradual decline in the last months.
3. KEEP THE SIGNAL NOISY - 30% of churners keep behaving normally ("silent churners") and 10% of
   retained customers have a temporary dip ("false alarms"), so the groups overlap.
4. DATA-QUALITY NOISE - the Kaggle file is already clean, but the project requires cleaning
   decisions with before/after evidence, so realistic export defects are injected into the RAW
   behaviour table (missing app tracking, duplicate rows, label variants, reversals, outliers,
   test accounts). The cleaning step (churnlib/cleaning.py) must find and treat them.
"""
import numpy as np
import pandas as pd

from .config import N_MONTHS, SEED, TXN_AMOUNT_COLS, TXN_COUNT_COLS

# --- generator parameters (documented assumptions) -------------------------------------------
P_DECLINE_CHURNER = 0.70      # share of churners who show a visible behavioural decline
P_DECLINE_RETAINED = 0.10     # share of retained customers with a temporary dip
DEPTH_CHURNER = (0.25, 0.65)  # relative drop reached in month 6 (uniform range)
DEPTH_RETAINED = (0.15, 0.45)
NOISE = {
    "missing_login_rate": 0.015,   # customers without app tracking
    "duplicate_rate": 0.005,       # rows exported twice
    "label_variant_rate": 0.02,    # inconsistent channel labels
    "negative_amount_rate": 0.002, # reversal artefacts (negative monthly amount)
    "extreme_amount_n": 30,        # data-entry errors (x20)
    "orphan_accounts": 25,         # internal test accounts not in the customer base
}


def _shares(age, mobile_pref, rng):
    """Channel shares (ATM, mobile, branch) from a Dirichlet draw whose mean depends on age."""
    branch_frac = np.clip(0.25 + 0.008 * (age - 18), 0.20, 0.70)
    atm_w = (1 - mobile_pref) * (1 - branch_frac)
    branch_w = (1 - mobile_pref) * branch_frac
    alpha = 8.0 * np.column_stack([atm_w, mobile_pref, branch_w])
    g = rng.gamma(alpha)
    s = g / g.sum(axis=1, keepdims=True)
    return s[:, 0], s[:, 1], s[:, 2]


def generate_behaviour(base: pd.DataFrame, seed: int = SEED, add_noise: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(base)
    age = base["age"].to_numpy(float)
    act = base["active_member"].to_numpy(float)
    prod = base["products_number"].to_numpy(float)
    sal = base["estimated_salary"].to_numpy(float)
    bal = base["balance"].to_numpy(float)
    churn = base["churn"].to_numpy(int)

    # ---- channel preference ------------------------------------------------------------
    mobile_pref = np.clip(0.75 - 0.011 * (age - 18) + rng.normal(0, 0.12, n), 0.05, 0.95)
    atm_share, mobile_share, branch_share = _shares(age, mobile_pref, rng)
    labels = np.array(["ATM", "Mobile app", "Branch"])
    preferred = labels[np.column_stack([atm_share, mobile_share, branch_share]).argmax(axis=1)]

    # ---- baseline monthly transaction rate ---------------------------------------------
    log_lam = (2.3 + 0.55 * act + 0.18 * (prod == 2) + 0.10 * (prod >= 3)
               - 0.008 * (age - 40) + 0.10 * (mobile_pref - 0.5) + rng.normal(0, 0.35, n))
    lam0 = np.exp(log_lam)

    # ---- disengagement multiplier matrix (n x 6) --------------------------------------
    decliner = np.where(churn == 1,
                        rng.random(n) < P_DECLINE_CHURNER,
                        rng.random(n) < P_DECLINE_RETAINED)
    depth = np.where(churn == 1,
                     rng.uniform(*DEPTH_CHURNER, n),
                     rng.uniform(*DEPTH_RETAINED, n))
    onset = rng.choice([3, 4], size=n)                       # month where the decline starts
    months = np.arange(1, N_MONTHS + 1)[None, :]
    ramp = np.clip((months - onset[:, None] + 1) / (N_MONTHS - onset[:, None] + 1), 0, 1)
    mult = 1 - (decliner * depth)[:, None] * ramp
    seasonal = rng.normal(1.0, 0.05, (n, N_MONTHS))

    counts = rng.poisson(lam0[:, None] * mult * seasonal)

    # ---- amounts: count x average ticket (right-skewed) -------------------------------
    ticket_mu = 35 + 0.00045 * sal + 0.00012 * bal
    ticket = rng.lognormal(np.log(ticket_mu), 0.45)
    jitter = rng.lognormal(0, 0.15, (n, N_MONTHS))
    amounts = np.round(counts * ticket[:, None] * jitter, 2)

    # ---- app logins (over-dispersed counts) -------------------------------------------
    log_mu = 1.9 + 0.8 * act + 1.2 * (mobile_pref - 0.4) - 0.01 * (age - 40)
    mu = np.exp(log_mu) * np.power(mult[:, -1], 1.3)
    r = 3.0
    logins = rng.poisson(rng.gamma(shape=r, scale=mu / r))
    days_login = np.where(
        logins == 0,
        rng.integers(30, 121, n),
        np.minimum(30, rng.exponential(30.0 / (logins + 1)).astype(int)),
    )

    # ---- recency of last transaction (consistent with the last monthly count) ---------
    c6 = counts[:, -1]
    days_txn = np.where(
        c6 == 0,
        np.minimum(180, 30 + rng.exponential(25, n).astype(int)),
        np.minimum(29, rng.exponential(30.0 / (c6 + 1)).astype(int)),
    )

    # ---- complaints ---------------------------------------------------------------------
    complaints = np.minimum(10, rng.poisson(0.15 + 0.45 * churn + 0.12 * (prod >= 3) + 0.05 * (1 - act)))

    df = pd.DataFrame({"customer_id": base["customer_id"].to_numpy()})
    for i in range(N_MONTHS):
        df[TXN_COUNT_COLS[i]] = counts[:, i].astype(int)
    for i in range(N_MONTHS):
        df[TXN_AMOUNT_COLS[i]] = amounts[:, i]
    df["login_count_30d"] = logins.astype(float)
    df["days_since_last_login"] = days_login.astype(float)
    df["days_since_last_txn"] = days_txn.astype(int)
    df["complaints_6m"] = complaints.astype(int)
    df["atm_share"] = np.round(atm_share, 4)
    df["mobile_share"] = np.round(mobile_share, 4)
    df["branch_share"] = np.round(1 - df["atm_share"] - df["mobile_share"], 4)
    df["preferred_channel"] = preferred

    return inject_export_defects(df, rng) if add_noise else df


def inject_export_defects(df: pd.DataFrame, rng) -> pd.DataFrame:
    """Simulate realistic defects of a raw system export (documented in the data dictionary)."""
    df = df.copy()
    n = len(df)

    # 1) missing app tracking
    miss = rng.random(n) < NOISE["missing_login_rate"]
    df.loc[miss, ["login_count_30d", "days_since_last_login"]] = np.nan

    # 2) inconsistent channel labels
    variants = {"ATM": ["atm", "ATM "], "Mobile app": ["mobile app", "MOBILE APP", "Mobile"],
                "Branch": ["branch", "BRANCH "]}
    idx = np.where(rng.random(n) < NOISE["label_variant_rate"])[0]
    for i in idx:
        orig = df.at[df.index[i], "preferred_channel"]
        df.at[df.index[i], "preferred_channel"] = rng.choice(variants[orig])

    # 3) reversal artefacts: negative monthly amount
    neg = np.where(rng.random(n) < NOISE["negative_amount_rate"])[0]
    for i in neg:
        col = rng.choice(TXN_AMOUNT_COLS)
        v = df.at[df.index[i], col]
        if v > 0:
            df.at[df.index[i], col] = -v

    # 4) extreme data-entry errors (x20)
    ext = rng.choice(n, size=NOISE["extreme_amount_n"], replace=False)
    for i in ext:
        col = rng.choice(TXN_AMOUNT_COLS)
        df.at[df.index[i], col] = round(df.at[df.index[i], col] * 20, 2)

    # 5) internal test accounts that are not customers
    orphans = df.sample(NOISE["orphan_accounts"], random_state=7).copy()
    orphans["customer_id"] = 99_000_000 + np.arange(len(orphans))
    df = pd.concat([df, orphans], ignore_index=True)

    # 6) rows exported twice
    dups = df.sample(int(NOISE["duplicate_rate"] * n), random_state=11)
    df = pd.concat([df, dups], ignore_index=True)

    return df.sample(frac=1, random_state=3).reset_index(drop=True)
