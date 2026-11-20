# Customer Churn Prediction & Segmentation for a Digital Bank

End-to-end analytics project: raw data → synthetic behaviour → EDA → cleaning → features → two models →
API + web app → monitoring → business recommendations. Built from the Business Analytics project announcement.

> **Data:** the public Kaggle *Bank Customer Churn Prediction* file (10,000 customers) plus a **synthetic behavioural layer**
> generated in Python. No real customers. All euro figures use **assumed** business parameters (`churnlib/config.py`).

## Results at a glance

| | |
|---|---|
| Final model | XGBoost, tuned with 5-fold CV (no resampling: class weights and SMOTE did not help) |
| Test ROC-AUC | **0.926** (CV 0.920) — logistic regression 0.873, random forest 0.912 |
| Profile-only vs + behaviour | 0.867 → 0.926 (the synthetic layer helps but does not make the task trivial) |
| Decision threshold | **0.08**, chosen by maximising net benefit with the cost rule → recall 94%, precision 42% |
| At threshold 0.5 | recall 67%, precision 80% |
| Risk bands (test) | observed churn: **High 80%**, Medium 19%, Low 2% |
| Segments (k = 5) | Dormant & Fading (59% churn, 4% of customers), Branch-Oriented Seniors (43%), High-Balance Savers (17%), Everyday Low-Balance Users (14%), Digital Power Users (7%) |
| Net benefit on test (assumed economics) | €131k with the model vs €99k contacting everybody and €108k at threshold 0.5 |
| Tests | 7 automated tests (API, validation, train/serve consistency) pass |

## How the 8 modules map to the code

| Module | Script / library | Main outputs |
|---|---|---|
| 1 Business understanding & data generation | `pipeline/01_generate_data.py`, `churnlib/synth.py`, `churnlib/dictionary.py` | `docs/data_dictionary.md/.csv`, `reports/synthetic_validation.md` |
| 2 EDA | `pipeline/02_eda.py` | `reports/eda_summary.md`, `reports/figures/eda_*.png` |
| 3 Cleaning | `pipeline/03_clean_data.py`, `churnlib/cleaning.py` | `reports/cleaning_log.md` (before/after), `data/processed/customers_clean.csv` |
| 4 Feature engineering | `pipeline/04_feature_engineering.py`, `churnlib/features.py` | `models/feature_config.json`, `reports/imbalance_comparison.csv`, `reports/feature_ranking.csv` |
| 5A Classification | `pipeline/05_train_classifier.py`, `churnlib/modeling.py` | `models/churn_pipeline.joblib`, `reports/model_results.md` |
| 5B Segmentation | `pipeline/06_segmentation.py`, `churnlib/segmentation.py` | `models/segmenter.joblib`, `models/personas.json`, `reports/segmentation_results.md` |
| 6 Deployment | `app/api.py` (FastAPI), `streamlit_app.py`, `churnlib/service.py` | live API + dashboard |
| 7 Monitoring | `pipeline/07_monitoring.py`, `churnlib/monitoring.py` | `reports/monitoring/` (Evidently HTML, summary), `docs/monitoring_triggers.md` |
| 8 Business translation | `pipeline/08_final_summary.py` | `reports/business_impact.md`, `reports/RESULTS_FOR_REPORT.md` |

`reports/RESULTS_FOR_REPORT.md` gathers every number for the written report (tables, metrics, segment sizes, monitoring).

## Run it

```bash
pip install -r requirements-pipeline.txt
python run_all.py                    # whole pipeline + tests (~10-15 min on 1 CPU; step 05 dominates)

uvicorn app.api:app --reload         # API  -> http://localhost:8000/docs
streamlit run streamlit_app.py       # app  -> http://localhost:8501
API_URL=http://localhost:8000 streamlit run streamlit_app.py   # app scoring through the API
python -m pytest tests -q
```

Trained artefacts are already in `models/`, so the API and the app run without re-training.

### API example

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "credit_score": 619, "country": "Germany", "gender": "Female", "age": 52, "tenure": 4, "balance": 120000,
  "products_number": 3, "credit_card": 1, "active_member": 0, "estimated_salary": 85000,
  "txn_counts": [14,13,12,8,6,4], "txn_amounts": [1200,1100,1000,700,520,300],
  "login_count_30d": 2, "days_since_last_login": 21, "days_since_last_txn": 24, "complaints_6m": 2,
  "atm_share": 0.35, "mobile_share": 0.15, "branch_share": 0.50}'
```
Endpoints: `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`, `GET /customer/{id}`. Inputs are validated (ranges, six monthly values, channel shares sum to 1).

## Deploy (free tiers)

The app needs these folders in the repository: `churnlib/`, `app/`, `models/`, `reports/`, `docs/`, `data/processed/` (`customers_clean.csv`, `test.csv`), `streamlit_app.py`, `requirements.txt`. Push the project to GitHub, then choose one:

1. **Streamlit Community Cloud (dashboard):** share.streamlit.io → *New app* → pick the repo → main file `streamlit_app.py`, Python 3.12.
2. **Hugging Face Spaces (dashboard):** create a *Streamlit* Space, upload the project, copy `deploy/huggingface_README.md` to the root as `README.md`.
3. **Render (API and dashboard):** *New → Blueprint* → pick the repo; `render.yaml` creates `churn-api` (health check `/health`) and `churn-dashboard`. To make the dashboard call the API, set `API_URL` to the API's public URL.
4. **Docker:** `docker build -t churn .` then `docker run -p 8000:8000 churn` (API) or override the command to serve Streamlit (see the Dockerfile comments).

Free-tier notes: services sleep after inactivity (first request is slow); set `LOG_PREDICTIONS=0` where the file system is read-only (already set in `render.yaml`).

## Key design decisions

* **No train/serve skew:** one `FeatureBuilder` and one set of cleaning rules are used for training and serving; a test checks the service reproduces the training pipeline's probabilities.
* **Leakage control:** split before fitting anything; encoders, scalers, RFM quintile edges and SMOTE are fitted inside training folds only.
* **Business-driven threshold:** contact when `a·(p·CLV − incentive) > contact cost`; the global threshold maximises realised net benefit on out-of-fold training predictions and is then evaluated once on the test set.
* **Explanations without extra dependencies:** for each group of inputs, the average of "remove it from this customer" and "add it to a typical customer" (a two-point approximation of Shapley values) — model-agnostic and not saturated for very high-risk customers.
* **Monitoring that catches both kinds of drift:** PSI/KS for inputs, AUC/recall for performance (labels arrive late), data-quality blocks, and explicit retraining triggers (`docs/monitoring_triggers.md`).

## Limitations (please state them in the report)

* The behavioural layer is **synthetic**, and churners were generated to decline in activity. Part of the lift from 0.867 to 0.926 AUC therefore reflects my assumptions, not real-world evidence. Realism checks are in `reports/synthetic_validation.md`.
* The Kaggle file is already clean, so the data-quality defects in the raw behaviour table were **injected on purpose** (documented in the data dictionary) to exercise the cleaning step.
* The Kaggle `churn` flag is interpreted as "left in the quarter after the snapshot" (an assumption, not a fact about the data).
* CLV, incentive, contact cost and acceptance rate are assumptions; the euro results illustrate the method, not a forecast. Impact of the recommended actions was **not** tested in an experiment.
* Monitoring is demonstrated on **simulated** months; thresholds need recalibration on real data.
* `gender` and `country` are model inputs; a real deployment needs a fairness review.
* The API has no authentication, rate limiting or CI/CD — it is a prototype.
