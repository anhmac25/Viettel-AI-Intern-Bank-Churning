"""Data dictionary content. Rendered to docs/data_dictionary.md and docs/data_dictionary.csv."""
import pandas as pd

COLS = ["column", "layer", "type", "unit", "valid_range", "generation_logic_or_business_meaning"]

ROWS = [
    # ------------------------------------------------------------------ Kaggle base layer
    ["customer_id", "Kaggle", "int", "-", "unique", "Unique customer key. Used only to join tables; never a model input."],
    ["credit_score", "Kaggle", "int", "points", "300-850", "Credit score of the customer."],
    ["country", "Kaggle", "category", "-", "France, Germany, Spain", "Country of residence."],
    ["gender", "Kaggle", "category", "-", "Female, Male", "Gender. Reported in EDA; see fairness note in README."],
    ["age", "Kaggle", "int", "years", "18-100", "Customer age."],
    ["tenure", "Kaggle", "int", "years", "0-10", "Years as a customer."],
    ["balance", "Kaggle", "float", "EUR", ">= 0", "Account balance. Zero is a real, frequent value (dormant account), not missing."],
    ["products_number", "Kaggle", "int", "count", "1-4", "Number of bank products held."],
    ["credit_card", "Kaggle", "binary", "-", "0/1", "1 if the customer holds a credit card."],
    ["active_member", "Kaggle", "binary", "-", "0/1", "1 if flagged as an active member."],
    ["estimated_salary", "Kaggle", "float", "EUR/year", ">= 0", "Estimated annual salary."],
    ["churn", "Kaggle", "binary", "-", "0/1", "TARGET. Interpreted as 'left the bank in the quarter after the snapshot date' (time assumption below)."],
    # ------------------------------------------------------------------ synthetic layer
    ["txn_count_m1 ... txn_count_m6", "Synthetic", "int", "transactions / month", "0-200",
     "m1 = oldest month, m6 = most recent. Poisson draw around a customer baseline: higher for active members and customers with 2 products, "
     "lower with age, slightly higher for mobile-preferring customers. 70% of churners (10% of retained customers) get a gradual decline that starts "
     "in month 3 or 4 and reaches a drop of 25-65% (churners) or 15-45% (retained) in month 6."],
    ["txn_amount_m1 ... txn_amount_m6", "Synthetic", "float", "EUR / month", ">= 0 after cleaning",
     "Count x average ticket x monthly jitter. Average ticket is log-normal (sigma 0.45) around 35 + 0.00045*salary + 0.00012*balance, giving right-skewed spending. "
     "RAW table contains defects (negative reversals, x20 entry errors)."],
    ["login_count_30d", "Synthetic", "float", "logins / 30 days", "0-150",
     "Negative-binomial draw (over-dispersed). Higher for active and mobile-preferring customers, lower with age; scaled down by the month-6 decline factor. "
     "RAW table has ~1.5% missing (customers without app tracking)."],
    ["days_since_last_login", "Synthetic", "float", "days", "0-120",
     "Exponential draw conditioned on login count; customers with 0 logins in 30 days get 30-120 days. Missing together with login_count_30d."],
    ["days_since_last_txn", "Synthetic", "int", "days", "0-180",
     "Derived from the month-6 transaction count so it is consistent with it (0 transactions -> at least 30 days)."],
    ["complaints_6m", "Synthetic", "int", "complaints", "0-10",
     "Poisson with rate 0.15 + 0.45*churn + 0.12*(products>=3) + 0.05*(inactive): long tail, higher for churners and multi-product customers."],
    ["atm_share, mobile_share, branch_share", "Synthetic", "float", "share of transactions", "0-1, sum = 1",
     "Dirichlet draw. Younger customers: mobile-heavy; older customers: branch- and ATM-heavy."],
    ["preferred_channel", "Synthetic", "category", "-", "ATM, Mobile app, Branch",
     "Channel with the highest share. RAW table has ~2% inconsistent labels (case, spaces, 'Mobile')."],
    # ------------------------------------------------------------------ engineered layer
    ["txn_freq_6m", "Engineered (F of RFM)", "int", "transactions", ">= 0", "Sum of the six monthly transaction counts."],
    ["txn_amount_6m", "Engineered (M of RFM)", "float", "EUR", ">= 0", "Sum of the six monthly amounts."],
    ["avg_ticket", "Engineered", "float", "EUR / transaction", ">= 0", "txn_amount_6m / max(txn_freq_6m, 1)."],
    ["recency_days", "Engineered (R of RFM)", "int", "days", "0-180", "= days_since_last_txn."],
    ["txn_ratio_recent", "Engineered", "float", "ratio", "> 0", "(mean count months 4-6 + 1) / (mean count months 1-3 + 1). Below 1 = declining activity."],
    ["txn_slope", "Engineered", "float", "per month", "any", "OLS slope of the six monthly counts divided by (mean + 1)."],
    ["amount_ratio_recent", "Engineered", "float", "ratio", "> 0", "Same idea as txn_ratio_recent but on amounts."],
    ["login_missing_flag", "Engineered", "binary", "-", "0/1", "1 if login data were missing in the raw export (imputed)."],
    ["complaint_flag", "Engineered", "binary", "-", "0/1", "1 if complaints_6m >= 1."],
    ["balance_to_salary", "Engineered", "float", "ratio", ">= 0", "balance / (estimated_salary + 1)."],
    ["products_per_tenure", "Engineered", "float", "products / year", ">= 0", "products_number / (tenure + 1)."],
    ["zero_balance_flag", "Engineered", "binary", "-", "0/1", "1 if balance == 0."],
    ["R_score, F_score, M_score, rfm_score", "Engineered", "int", "quintile 1-5", "1-5 (rfm 3-15)",
     "Quintile scores using edges fitted on the TRAINING set (recency reversed: recent = 5). rfm_score = R + F + M."],
]


def dictionary_df() -> pd.DataFrame:
    return pd.DataFrame(ROWS, columns=COLS)


TIME_ASSUMPTION = (
    "**Time assumption.** Each customer has a six-month observation window ending on a snapshot date. The Kaggle `churn` flag is interpreted as "
    "'left the bank in the quarter after the snapshot'. All behavioural fields are generated inside the window, so no behavioural field contains "
    "information from after the snapshot."
)

BUSINESS_NOTE = (
    "**Business assumptions (replace with real figures).** CLV proxy = (0.5% x balance + 25 EUR x products) x 3 years. Contact cost 10 EUR, incentive 100 EUR, "
    "30% acceptance. Expected value of contacting a customer = a x (p x CLV - incentive) - contact cost, so a customer is worth contacting when "
    "p > (incentive + contact_cost / a) / CLV."
)


def to_markdown() -> str:
    df = dictionary_df()
    out = ["# Data Dictionary", "",
           "Every field of the combined dataset: the public Kaggle profile layer, the synthetic behavioural layer generated in Python "
           "(`churnlib/synth.py`, seed 42) and the engineered features (`churnlib/features.py`).", "",
           TIME_ASSUMPTION, "", BUSINESS_NOTE, "",
           "| " + " | ".join(COLS) + " |", "|" + "---|" * len(COLS)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(str(v).replace("|", "/") for v in r.tolist()) + " |")
    out += ["", "## Simulated data-quality defects in the RAW behaviour table", "",
            "The Kaggle file has no missing values or duplicates, but the project requires cleaning decisions with before/after evidence. "
            "The raw behaviour export therefore contains realistic defects: ~1.5% missing login data, ~0.5% duplicated rows, ~2% inconsistent channel labels, "
            "~0.2% negative monthly amounts (reversals), 30 extreme amounts (x20 entry errors) and 25 internal test accounts that are not customers. "
            "`churnlib/cleaning.py` detects and treats each of them and writes a cleaning log (`reports/cleaning_log.csv`)."]
    return "\n".join(out)
