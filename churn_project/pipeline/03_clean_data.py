"""Module 3 - Data cleaning. Merges the Kaggle profile table with the raw behaviour table, treats every
issue found and documents it with before/after evidence.

Outputs
  data/processed/customers_clean.csv
  reports/cleaning_log.csv / cleaning_log.md
  models/cleaning_params.json   (used again at serving time)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from churnlib import config as C
from churnlib.cleaning import merge_and_clean, save_params

base = pd.read_csv(C.BASE_FILE)
raw = pd.read_csv(C.BEHAVIOUR_RAW)

clean, log, params = merge_and_clean(base, raw)
clean.to_csv(C.MERGED_CLEAN, index=False)
save_params(params, C.MODELS / "cleaning_params.json")
log.to_csv(C.REPORTS / "cleaning_log.csv", index=False)

md = ["# Cleaning log (before / after)", "", log.to_markdown(index=False), "",
      f"Final dataset: {clean.shape[0]} customers x {clean.shape[1]} columns, no missing values, unique customer_id.",
      f"Churn rate after cleaning: {clean.churn.mean():.4f}"]
(C.REPORTS / "cleaning_log.md").write_text("\n".join(md), encoding="utf-8")
print(log.to_string(index=False))
print(clean.shape, "churn", round(clean.churn.mean(), 4))
