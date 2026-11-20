# Numbers for the written report

Every figure below was produced by the pipeline; copy them into the yellow placeholders of the Word report.

## Dataset

- customers: 10,000; columns after merge: 33; overall churn rate: 20.37%
- train / test: 8000 / 2000 (stratified 80/20, seed 42); train churn rate 20.38%
- zero-balance customers: 36.2%

## EDA (Table 3.4 of the report)

See reports/eda_summary.md - churn rates by country, age band, products, activity, gender.

## Cleaning log (Table 3.5)

| issue                                             | how_detected                                                                    | treatment                                                                      |   before |   after |
|:--------------------------------------------------|:--------------------------------------------------------------------------------|:-------------------------------------------------------------------------------|---------:|--------:|
| Missing values (Kaggle)                           | isna().sum()                                                                    | none needed                                                                    |        0 |       0 |
| Duplicate customers (Kaggle)                      | duplicated(customer_id)                                                         | none needed                                                                    |        0 |       0 |
| Invalid ages / balances / scores (Kaggle)         | range rules: age 18-100, balance >= 0, score 300-850, tenure 0-10, products 1-4 | drop invalid rows                                                              |        0 |       0 |
| Inconsistent categories (Kaggle)                  | unique values of country, gender                                                | none needed                                                                    |        5 |       5 |
| Duplicate behaviour rows                          | duplicated() on all columns                                                     | keep first occurrence                                                          |       50 |       0 |
| Duplicate customer_id after de-duplication        | duplicated(customer_id)                                                         | keep first                                                                     |        0 |       0 |
| Ids not in the customer base (test accounts)      | anti-join on customer_id                                                        | drop                                                                           |       25 |       0 |
| Inconsistent channel labels                       | unique values of preferred_channel                                              | strip + lower-case + map to 3 canonical labels                                 |       10 |       3 |
| Customers without behavioural history after merge | inner join size vs base                                                         | none (all customers matched)                                                   |        0 |       0 |
| Negative monthly amounts (reversal artefacts)     | amount < 0                                                                      | set to missing, impute with the customer's own median month                    |       23 |       0 |
| Extreme monthly amounts (x20 entry errors)        | amount > 99.5th percentile of positive amounts (8478 EUR)                       | winsorise at the 99.5th percentile; original values not kept in modelling file |      300 |       0 |
| Missing login data (no app tracking)              | isna() on login_count_30d                                                       | impute median by activity status + login_missing_flag                          |      131 |       0 |

## Class-imbalance experiment (5-fold CV on train)

| model   | imbalance    |   roc_auc |   pr_auc |   recall |   precision |   brier |
|:--------|:-------------|----------:|---------:|---------:|------------:|--------:|
| logreg  | none         |    0.8686 |   0.7067 |   0.5006 |      0.748  |  0.1005 |
| logreg  | class_weight |    0.8688 |   0.7031 |   0.7644 |      0.5091 |  0.141  |
| logreg  | smote        |    0.8668 |   0.7009 |   0.762  |      0.5093 |  0.1405 |
| rf      | none         |    0.9029 |   0.776  |   0.5613 |      0.8383 |  0.0888 |
| rf      | class_weight |    0.9034 |   0.7674 |   0.6699 |      0.73   |  0.0963 |
| rf      | smote        |    0.9033 |   0.7613 |   0.6411 |      0.7581 |  0.0935 |
| xgb     | none         |    0.9183 |   0.8089 |   0.6374 |      0.8169 |  0.0784 |
| xgb     | class_weight |    0.9169 |   0.8063 |   0.7883 |      0.6497 |  0.0963 |
| xgb     | smote        |    0.9174 |   0.8083 |   0.6712 |      0.776  |  0.0797 |

Chosen strategy: none for the final model. SMOTE and class weights did not improve ROC-AUC enough to justify distorting the predicted probabilities.

## Model comparison (Table 3.7)

| model   |   cv_roc_auc |   test_roc_auc |   test_pr_auc |   test_brier |   test_precision |   test_recall |   test_f1 |
|:--------|-------------:|---------------:|--------------:|-------------:|-----------------:|--------------:|----------:|
| logreg  |       0.8687 |         0.8727 |        0.6963 |       0.1026 |           0.7108 |        0.5012 |    0.5879 |
| rf      |       0.9042 |         0.9121 |        0.7985 |       0.0868 |           0.831  |        0.5799 |    0.6831 |
| xgb     |       0.92   |         0.9258 |        0.8193 |       0.0763 |           0.7988 |        0.6732 |    0.7307 |

- final model: **xgb**, CV ROC-AUC 0.9200
- at business threshold 0.08: ROC-AUC 0.926, PR-AUC 0.819, precision 0.415, recall 0.936, F1 0.575
- at threshold 0.5: precision 0.799, recall 0.673, F1 0.731
- profile-only AUC 0.867 vs profile + behaviour 0.926
- out-of-fold Brier score 0.0780

## Segments (Table 3.8) and segment impact

| persona                    |   customers |   churn rate % |   share flagged % |   churners reached % |   avg CLV (EUR) |   net benefit of contacting flagged (EUR) |
|:---------------------------|------------:|---------------:|------------------:|---------------------:|----------------:|------------------------------------------:|
| Dormant & Fading           |          68 |           58.8 |              79.4 |                 97.5 |            1338 |                                     13375 |
| Branch-Oriented Seniors    |         369 |           44.4 |              78   |                 98.2 |            1467 |                                     65062 |
| High-Balance Savers        |         675 |           15   |              52.6 |                 90.1 |            1935 |                                     40135 |
| Everyday Low-Balance Users |         507 |           15.8 |              32   |                 95   |             212 |                                      7184 |
| Digital Power Users        |         381 |            5.8 |              15.5 |                 63.6 |            1289 |                                      5104 |

## Observed churn by risk band

| band   |   customers |   observed churn rate % |
|:-------|------------:|------------------------:|
| High   |         343 |                    79.9 |
| Medium |         575 |                    18.6 |
| Low    |        1082 |                     2.4 |

## Monitoring (Table 3.10 and the simulation)

| month   | scenario                         |    auc |   recall | input_drift   | prediction_drift   | performance   | data_quality   | action                                               |
|:--------|:---------------------------------|-------:|---------:|:--------------|:-------------------|:--------------|:---------------|:-----------------------------------------------------|
| Month 1 | Stable                           | 0.9175 |   0.9274 | OK            | OK                 | OK            | OK             | No action                                            |
| Month 2 | Mild shift                       | 0.9372 |   0.9464 | OK            | OK                 | OK            | OK             | No action                                            |
| Month 3 | App outage / activity drop       | 0.8872 |   0.9781 | ALERT         | WARNING            | OK            | OK             | Schedule retraining and investigate the data source  |
| Month 4 | Competitor offer (concept drift) | 0.7442 |   0.677  | OK            | OK                 | WARNING       | OK             | Investigate (data change, segment mix, labels)       |
| Month 5 | Login tracking failure           | 0.9049 |   0.9137 | OK            | OK                 | OK            | BLOCK          | Block scoring of this batch and alert the data owner |
| Month 6 | Competitor offer continues       | 0.7367 |   0.6511 | OK            | OK                 | ALERT         | OK             | RETRAIN and revalidate before replacing the model    |

## Where the other results live

- segmentation details and the segment-as-feature test: reports/segmentation_results.md
- model details and strategy comparison: reports/model_results.md
- business impact by segment: reports/business_impact.md
- deployed URLs: fill in after deployment (README, section 'Deploy').