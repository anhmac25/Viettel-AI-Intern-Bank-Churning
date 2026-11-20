"""Central configuration: paths, random seed, column groups and business assumptions."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INT = ROOT / "data" / "interim"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
MONITOR = REPORTS / "monitoring"
DOCS = ROOT / "docs"
LOGS = ROOT / "logs"

BASE_FILE = DATA_RAW / "Bank_Customer_Churn_Prediction.csv"
BEHAVIOUR_RAW = DATA_INT / "behaviour_raw.csv"
MERGED_CLEAN = DATA_PROC / "customers_clean.csv"
FEATURES_FILE = DATA_PROC / "features_all.csv"
TRAIN_FILE = DATA_PROC / "train.csv"
TEST_FILE = DATA_PROC / "test.csv"

SEED = 42
N_MONTHS = 6
MODEL_VERSION = "1.0.0"

# ---------------------------------------------------------------- columns
TARGET = "churn"
ID_COL = "customer_id"
PROFILE_COLS = [
    "credit_score", "country", "gender", "age", "tenure", "balance",
    "products_number", "credit_card", "active_member", "estimated_salary",
]
TXN_COUNT_COLS = [f"txn_count_m{i}" for i in range(1, N_MONTHS + 1)]
TXN_AMOUNT_COLS = [f"txn_amount_m{i}" for i in range(1, N_MONTHS + 1)]
BEHAVIOUR_COLS = (
    TXN_COUNT_COLS + TXN_AMOUNT_COLS
    + ["login_count_30d", "days_since_last_login", "days_since_last_txn",
       "complaints_6m", "atm_share", "mobile_share", "branch_share", "preferred_channel"]
)
# columns a caller must provide to score one customer
INPUT_COLS = PROFILE_COLS + BEHAVIOUR_COLS

CHANNELS = ["ATM", "Mobile app", "Branch"]

# ---------------------------------------------------------------- business assumptions
# All values are ASSUMPTIONS (the data are public/synthetic) and are meant to be replaced by
# the bank's real figures. They drive the CLV proxy, the decision threshold and the app.
BUSINESS = {
    "margin_on_balance": 0.005,   # annual net margin earned on the customer's balance
    "margin_per_product": 25.0,   # annual net margin per product held (EUR)
    "lifetime_years": 3.0,        # expected remaining relationship if retained
    "contact_cost": 10.0,         # cost of one retention contact (EUR)
    "incentive": 100.0,           # cost of the offer if the customer accepts (EUR)
    "accept_rate": 0.30,          # share of contacted customers who accept the offer
}


def clv_proxy(balance, products_number):
    """Customer lifetime value proxy in EUR (documented assumption, see docs)."""
    b = BUSINESS
    annual = b["margin_on_balance"] * balance + b["margin_per_product"] * products_number
    return annual * b["lifetime_years"]


def expected_value_of_contact(p_churn, clv):
    """EV(contact) = a * (p * CLV - incentive) - contact_cost  (derivation in docs)."""
    b = BUSINESS
    return b["accept_rate"] * (p_churn * clv - b["incentive"]) - b["contact_cost"]
