"""Re-run the whole pipeline, then the tests. Usage:  python run_all.py   (about 10-15 min on one CPU; step 05 dominates)"""
import subprocess
import sys

STEPS = ["01_generate_data", "02_eda", "03_clean_data", "04_feature_engineering", "05_train_classifier",
         "06_segmentation", "07_monitoring", "08_final_summary"]
for s in STEPS:
    print(f"\n=== {s} ===", flush=True)
    subprocess.run([sys.executable, f"pipeline/{s}.py"], check=True)
print("\n=== tests ===", flush=True)
subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], check=True)
