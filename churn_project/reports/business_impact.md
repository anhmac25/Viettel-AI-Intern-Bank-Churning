# Business impact (held-out test set)

Assumptions (replace with the bank's figures): {'margin_on_balance': 0.005, 'margin_per_product': 25.0, 'lifetime_years': 3.0, 'contact_cost': 10.0, 'incentive': 100.0, 'accept_rate': 0.3}. Test customers: 2000.

## Observed churn by risk band (the bands are well calibrated)

| band   |   customers |   observed churn rate % |
|:-------|------------:|------------------------:|
| High   |         343 |                    79.9 |
| Medium |         575 |                    18.6 |
| Low    |        1082 |                     2.4 |

## Impact by segment when contacting every customer with churn probability >= 0.08

| persona                    |   customers |   churn rate % |   share flagged % |   churners reached % |   avg CLV (EUR) |   net benefit of contacting flagged (EUR) |
|:---------------------------|------------:|---------------:|------------------:|---------------------:|----------------:|------------------------------------------:|
| Dormant & Fading           |          68 |           58.8 |              79.4 |                 97.5 |            1338 |                                     13375 |
| Branch-Oriented Seniors    |         369 |           44.4 |              78   |                 98.2 |            1467 |                                     65062 |
| High-Balance Savers        |         675 |           15   |              52.6 |                 90.1 |            1935 |                                     40135 |
| Everyday Low-Balance Users |         507 |           15.8 |              32   |                 95   |             212 |                                      7184 |
| Digital Power Users        |         381 |            5.8 |              15.5 |                 63.6 |            1289 |                                      5104 |

Total net benefit of the model-based policy on the test set: **EUR 130,861**. Policy 'contact when expected value of contact > 0 for that customer' gives EUR 138,394.

Note: the euro figures depend entirely on the assumed CLV, incentive, contact cost and acceptance rate, and on the synthetic behaviour; they illustrate the method, they are not a forecast.