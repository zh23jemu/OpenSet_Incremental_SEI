# MV-ACC-CIL End-to-End Result (ManyTx RX index 2)

This run uses the neural classifier directly after each class-head expansion; it does **not** use prototype matching.

| Round | Clusters | NMI | ARI | Overall Acc | New Acc | Forgetting |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 10 | 0.9178 | 0.8907 | 0.3033 | 0.5067 | 0.3267 |
| R2 | 11 | 0.9466 | 0.9202 | 0.2333 | 0.4867 | 0.4267 |
| R3 | 10 | 0.8929 | 0.8117 | 0.2700 | 0.5600 | 0.4267 |

Discovery has 100% assignment coverage, but the Day1 neural classifier itself is weak on held-out ManyTx samples (initial accuracy 0.4267). Therefore this dataset currently demonstrates a discovery-versus-end-to-end-classification gap, not a successful final CIL result. Raw tables are in `clustering_results.csv` and `incremental_results.csv`.
