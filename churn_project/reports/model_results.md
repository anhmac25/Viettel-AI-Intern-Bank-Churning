# Churn classifier - results

Final model: **xgb** (imbalance strategy: none), model version 1.0.0

## Model comparison (tuned with 5-fold CV on train; test metrics at threshold 0.5)

| model   |   cv_roc_auc |   test_roc_auc |   test_pr_auc |   test_brier |   test_precision |   test_recall |   test_f1 |
|:--------|-------------:|---------------:|--------------:|-------------:|-----------------:|--------------:|----------:|
| logreg  |       0.8687 |         0.8727 |        0.6963 |       0.1026 |           0.7108 |        0.5012 |    0.5879 |
| rf      |       0.9042 |         0.9121 |        0.7985 |       0.0868 |           0.831  |        0.5799 |    0.6831 |
| xgb     |       0.92   |         0.9258 |        0.8193 |       0.0763 |           0.7988 |        0.6732 |    0.7307 |

## Best hyper-parameters

- logreg: `{'C': 0.1}`
- rf: `{'n_estimators': 300, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'max_depth': None}`
- xgb: `{'subsample': 0.9, 'reg_lambda': 3, 'n_estimators': 600, 'min_child_weight': 1, 'max_depth': 3, 'learning_rate': 0.03, 'colsample_bytree': 1.0}`

## Test-set performance of the final model

|   threshold |   roc_auc |   pr_auc |   brier |   precision |   recall |     f1 |
|------------:|----------:|---------:|--------:|------------:|---------:|-------:|
|        0.5  |    0.9258 |   0.8193 |  0.0763 |      0.7988 |   0.6732 | 0.7307 |
|        0.08 |    0.9258 |   0.8193 |  0.0763 |      0.415  |   0.9361 | 0.5751 |

## Business decision threshold

Chosen on out-of-fold training predictions by maximising realised net benefit: **0.08**. Assumptions: {'margin_on_balance': 0.005, 'margin_per_product': 25.0, 'lifetime_years': 3.0, 'contact_cost': 10.0, 'incentive': 100.0, 'accept_rate': 0.3}.

| strategy                       |   customers_contacted |   churners_reached |   recall |   precision |   net_benefit_eur |
|:-------------------------------|----------------------:|-------------------:|---------:|------------:|------------------:|
| Contact nobody                 |                     0 |                  0 |    0     |       0     |               0   |
| Contact everybody              |                  2000 |                407 |    1     |       0.204 |           99002.5 |
| Model, threshold 0.5           |                   343 |                274 |    0.673 |       0.799 |          107945   |
| Model, business threshold 0.08 |                   918 |                381 |    0.936 |       0.415 |          130861   |

## Does the synthetic behavioural layer make the problem trivial?

- test ROC-AUC with profile columns only: **0.867**
- test ROC-AUC with profile + behaviour: **0.926**
- improvement: +0.058. The behavioural layer helps but does not make the problem trivial; because the layer was generated with a churn-linked decline, part of this gain reflects the generator's assumptions and must not be read as real-world evidence.

## Calibration

Out-of-fold Brier score: 0.0780 (always predicting the base rate gives 0.1622).