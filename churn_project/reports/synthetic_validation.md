# Synthetic data validation

## Automated checks (defect-free copy)

- PASS - No negative counts
- PASS - No negative amounts (defect-free copy)
- PASS - Channel shares sum to 1 (+/-0.001)
- PASS - 0 logins => >= 30 days since last login
- PASS - 0 transactions in m6 => >= 30 days since last txn
- PASS - Correlation signs match business expectation

## Correlations

- corr(txn frequency, logins): +0.345
- corr(complaints, churn): +0.353
- corr(recent/early txn ratio, churn): -0.292
- corr(logins, churn): -0.260
- corr(age, mobile share): -0.506

## Group means (0 = retained, 1 = churned)

|   churn |   freq6 |   ratio |   login_count_30d |   complaints_6m |   days_since_last_txn |
|--------:|--------:|--------:|------------------:|----------------:|----------------------:|
|       0 | 100.137 |   1.003 |            13.381 |           0.178 |                 1.849 |
|       1 |  70.951 |   0.829 |             6.532 |           0.678 |                 4.194 |

Churners and retained customers overlap by design (30% silent churners, 10% temporary dips); see reports/model_results.md for the profile-only vs combined model comparison that tests whether the synthetic layer makes the problem trivial.

## Defects injected in the raw table

- rows raw: 10075 vs customers: 10000
- missing login_count_30d: 131
- exact duplicate rows: 50
- distinct channel labels: 10
- negative monthly amounts: 23
- ids not in customer base: 25