# Monitoring signals and retraining triggers

| Signal | Metric | Warning | Alert / action |
|---|---|---|---|
| Input data drift | PSI per monitored feature (and KS test); top-5 features by model importance | PSI > 0.1 on a top feature | PSI > 0.25 on any top-5 feature, or > 30% of monitored features above 0.25 -> schedule retraining, investigate source |
| Prediction drift | relative change of the mean predicted probability vs reference | > 20% | investigate data change and segment mix |
| Model performance | AUC-ROC and recall at the business threshold (labels arrive after the prediction horizon) | AUC drop > 0.05 vs baseline, or recall < 85%, in one month | the same breach in 2 consecutive valid months -> RETRAIN and revalidate |
| Data quality | missing-value rate per field; schema check | - | missing > 5% or schema violation -> BLOCK the batch and alert the data owner (blocked batches are skipped in the consecutive-breach rule) |
| Scheduled review | calendar | - | retrain at least every 3 months (the prediction horizon is one quarter) |

PSI rule of thumb: < 0.10 stable, 0.10-0.25 moderate shift, > 0.25 significant shift. Thresholds are starting points and must be recalibrated on real data.

Why performance monitoring is required in addition to drift monitoring: in the simulation (reports/monitoring_results.md) the competitor-offer months change who churns without changing the inputs, so PSI stays near zero while AUC falls from about 0.93 to about 0.74.