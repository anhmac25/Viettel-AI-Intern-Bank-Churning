# Monitoring results (simulated months)

Reference: 8000 training customers; baseline test AUC 0.926; threshold 0.08.

| month | scenario | AUC | recall | input drift | prediction drift | performance | data quality | action |
|---|---|---|---|---|---|---|---|---|
| Month 1 | Stable | 0.917 | 0.927 | OK | OK | OK | OK | No action |
| Month 2 | Mild shift | 0.937 | 0.946 | OK | OK | OK | OK | No action |
| Month 3 | App outage / activity drop | 0.887 | 0.978 | ALERT | WARNING | OK | OK | Schedule retraining and investigate the data source |
| Month 4 | Competitor offer (concept drift) | 0.744 | 0.677 | OK | OK | WARNING | OK | Investigate (data change, segment mix, labels) |
| Month 5 | Login tracking failure | 0.905 | 0.914 | OK | OK | OK | BLOCK | Block scoring of this batch and alert the data owner |
| Month 6 | Competitor offer continues | 0.737 | 0.651 | OK | OK | ALERT | OK | RETRAIN and revalidate before replacing the model |