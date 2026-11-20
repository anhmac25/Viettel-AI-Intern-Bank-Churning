# Cleaning log (before / after)

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

Final dataset: 10000 customers x 33 columns, no missing values, unique customer_id.
Churn rate after cleaning: 0.2037