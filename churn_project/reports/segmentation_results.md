# Segmentation results

K-Means with **k = 5** on 9 standardised behavioural/RFM features (log transform on spending and balance); fitted on the training set; the churn label is NOT used.

## Choosing k

|   k |   inertia |   silhouette |
|----:|----------:|-------------:|
|   2 |     60727 |        0.144 |
|   3 |     54749 |        0.15  |
|   4 |     50132 |        0.143 |
|   5 |     46375 |        0.151 |
|   6 |     43884 |        0.147 |
|   7 |     41602 |        0.149 |
|   8 |     39689 |        0.144 |

Silhouette scores are low for every k (customer behaviour is a continuum, without sharp natural groups), so k was chosen for interpretability: k = 5 has the best (or tied-best) silhouette among k >= 4 and is the first solution that isolates a small, high-risk 'dormant & fading' group.

## Personas

| persona                    |   cluster |   share % |   churn % (all) |   churn % (train) |   churn % (test) | profile (vs average)                                                                                  |
|:---------------------------|----------:|----------:|----------------:|------------------:|-----------------:|:------------------------------------------------------------------------------------------------------|
| Dormant & Fading           |         4 |       3.6 |            59.3 |              59.5 |             58.8 | higher days since last transaction, age; lower recent activity trend, transaction frequency, spending |
| Branch-Oriented Seniors    |         0 |      17.8 |            42.8 |              42.4 |             44.4 | higher age; lower mobile-app usage, app logins, transaction frequency                                 |
| High-Balance Savers        |         1 |      33.4 |            16.9 |              17.3 |             15   | higher balance; lower number of products, age, transaction frequency                                  |
| Everyday Low-Balance Users |         3 |      25.6 |            14.1 |              13.7 |             15.8 | higher number of products; lower balance                                                              |
| Digital Power Users        |         2 |      19.7 |             7   |               7.3 |              5.8 | higher transaction frequency, spending, app logins; lower age, days since last transaction            |

## Average profile per persona (original units)

|                            |   recency_days |   txn_freq_6m |   txn_amount_6m |   login_count_30d |   mobile_share |   balance |   products_number |   txn_ratio_recent |   age |
|:---------------------------|---------------:|--------------:|----------------:|------------------:|---------------:|----------:|------------------:|-------------------:|------:|
| Dormant & Fading           |          18.53 |         46.46 |         4325.76 |              5.84 |           0.49 |  92887.2  |              1.38 |               0.72 | 42.75 |
| Branch-Oriented Seniors    |           2.15 |         77.01 |         7432.85 |              6.81 |           0.25 |  93571.1  |              1.41 |               0.91 | 54.09 |
| High-Balance Savers        |           2    |         77.25 |         7546.71 |             10.15 |           0.59 | 119578    |              1.25 |               1    | 35.22 |
| Everyday Low-Balance Users |           1.81 |         84.06 |         6758.47 |             10.09 |           0.55 |   3878.26 |              1.87 |               0.99 | 36.74 |
| Digital Power Users        |           0.76 |        159.99 |        17933.5  |             22.9  |           0.6  |  79090.5  |              1.69 |               0.99 | 33.88 |

## Segment as a classifier feature

5-fold CV ROC-AUC without segment: **0.9195**; with segment: **0.9197** (delta +0.0001). The segment adds little predictive power, because it is built from the same behaviour the classifier already sees - its value is interpretability and targeting, so the deployed classifier does not use it.

## Retention actions by persona and risk band

### Dormant & Fading

- **low:** Light-touch reactivation email.
- **medium:** Reactivation campaign with a simple first action (e.g. set up a direct debit) and a small incentive.
- **high:** Immediate outbound contact: understand the reason, resolve open complaints, offer a targeted win-back incentive.

### Branch-Oriented Seniors

- **low:** No contact. Offer help to use digital services at the next branch visit.
- **medium:** Friendly call or letter from the branch; guided digital onboarding.
- **high:** Personal call from the branch relationship manager within one week; priority service line and fee review.

### High-Balance Savers

- **low:** No contact. Annual relationship review.
- **medium:** Relationship-manager check-in; competitive savings or investment offer.
- **high:** Priority call by a senior relationship manager; bespoke retention offer protecting the balance.

### Everyday Low-Balance Users

- **low:** No contact; promote savings goals in the app.
- **medium:** Low-cost automated offer: fee waiver or salary-deposit bonus.
- **high:** Review products and fees (many products, low balance); proactive complaint resolution; simplify the product bundle.

### Digital Power Users

- **low:** No contact. Keep monitoring; cross-sell through in-app offers.
- **medium:** In-app nudge or cashback on their favourite spending category; avoid costly discounts.
- **high:** Alert the digital-engagement team: personalised in-app offer and a short survey about what changed.
