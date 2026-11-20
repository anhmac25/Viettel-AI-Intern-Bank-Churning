# EDA summary

Customers: 10,000 | overall churn rate: **20.4%**

## Churn rate by country

| country   |   rate |   customers |
|:----------|-------:|------------:|
| France    |   16.2 |        5014 |
| Germany   |   32.4 |        2509 |
| Spain     |   16.7 |        2477 |

## Churn rate by gender

| gender   |   rate |   customers |
|:---------|-------:|------------:|
| Female   |   25.1 |        4543 |
| Male     |   16.5 |        5457 |

## Churn rate by number of products

|   products_number |   rate |   customers |
|------------------:|-------:|------------:|
|                 1 |   27.7 |        5084 |
|                 2 |    7.6 |        4590 |
|                 3 |   82.7 |         266 |
|                 4 |  100   |          60 |

## Churn rate by active member

|   active_member |   rate |   customers |
|----------------:|-------:|------------:|
|               0 |   26.8 |        4849 |
|               1 |   14.3 |        5151 |

## Churn rate by age band

| age_band   |   rate |   customers |
|:-----------|-------:|------------:|
| 18-29      |    7.6 |        1641 |
| 30-39      |   10.9 |        4346 |
| 40-49      |   30.8 |        2618 |
| 50-59      |   56   |         869 |
| 60+        |   28   |         526 |

## Churn rate by tenure (years)

| tenure_band   |   rate |   customers |
|:--------------|-------:|------------:|
| 0-2           |   21.2 |        2496 |
| 3-5           |   20.8 |        3010 |
| 6-8           |   18.9 |        3020 |
| 9-10          |   21.3 |        1474 |

## Churn rate by balance band

| balance_band   |   rate |   customers |
|:---------------|-------:|------------:|
| zero           |   13.8 |        3617 |
| low            |   20.6 |        1596 |
| mid-low        |   27.4 |        1596 |
| mid-high       |   25.3 |        1595 |
| high           |   23   |        1596 |

## Churn rate by has credit card

|   credit_card |   rate |   customers |
|--------------:|-------:|------------:|
|             0 |   20.8 |        2945 |
|             1 |   20.2 |        7055 |

## Distribution notes

- 36.2% of customers hold a zero balance (a real dormant-account behaviour, kept as a feature).
- Age: median 37, max 92 (all within 18-100).
- Salary shows almost no difference between churners and retained customers (median 102,461 vs 99,645).

## Correlation with churn (Pearson)

|                     |   corr |
|:--------------------|-------:|
| complaints_6m       |  0.353 |
| age                 |  0.285 |
| login_count_30d     | -0.26  |
| txn_freq_6m         | -0.244 |
| days_since_last_txn |  0.216 |
| active_member       | -0.156 |
| mobile_share        | -0.142 |
| txn_amount_6m       | -0.127 |
| balance             |  0.119 |
| products_number     | -0.048 |
| credit_score        | -0.027 |
| tenure              | -0.014 |
| estimated_salary    |  0.012 |

## Behavioural trend (mean monthly transactions)

|   churn |   txn_count_m1 |   txn_count_m2 |   txn_count_m3 |   txn_count_m4 |   txn_count_m5 |   txn_count_m6 |
|--------:|---------------:|---------------:|---------------:|---------------:|---------------:|---------------:|
|       0 |          16.87 |          16.88 |          16.83 |          16.66 |          16.53 |          16.37 |
|       1 |          13.37 |          13.31 |          12.91 |          11.75 |          10.44 |           9.17 |

### Churn rate by Recency (5 = most recent)

|   R |   churn % |
|----:|----------:|
|   1 |      36   |
|   2 |      21.4 |
|   3 |      16.9 |
|   4 |      13.5 |

### Churn rate by Frequency (5 = most transactions)

|   F |   churn % |
|----:|----------:|
|   1 |      39.7 |
|   2 |      24.6 |
|   3 |      17.8 |
|   4 |      11.6 |
|   5 |       8.2 |

### Churn rate by Monetary (5 = highest amount)

|   M |   churn % |
|----:|----------:|
|   1 |      32.6 |
|   2 |      22.2 |
|   3 |      19.3 |
|   4 |      14.8 |
|   5 |      12.8 |

## Data-quality findings that feed Module 3

- missing login_count_30d: 131
- negative monthly amounts (raw): 23
- amounts above 99.5th percentile (raw): 303
- distinct channel labels: 10

## Key insights

1. **Country:** Germany churns at 32.4% vs 16.2% in France.
2. **Products:** 1 product 27.7%, 2 products 7.6%, 3 products 82.7%, 4 products 100.0% - two products is the healthy level.
3. **Activity:** inactive members churn at 26.8% vs 14.3% for active members.
4. **Age:** the highest-risk band is 50-59 (56.0%); the lowest is 18-29 (7.6%).
5. **Gender:** female customers churn at 25.1% vs 16.5% for male customers.
6. **Behaviour:** churners start at a lower level (13.4 vs 16.9 transactions in month 1, partly because they are less often active members) and then fall by 31% by month 6, while retained customers stay almost flat (3% change) - the TREND adds information beyond the level.
7. **Salary and credit card ownership** show almost no relationship with churn.