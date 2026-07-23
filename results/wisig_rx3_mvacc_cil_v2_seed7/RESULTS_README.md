# MV-ACC-CIL End-to-End Result (WiSig RX3)

This run uses the neural classifier directly after each class-head expansion; it does **not** use prototype matching.

| Round | Clusters | NMI | ARI | Overall Acc | New Acc | Forgetting |
|---|---:|---:|---:|---:|---:|---:|
| R1 | 10 | 0.9071 | 0.8817 | 0.9006 | 0.9289 | 0.1256 |
| R2 | 10 | 0.9633 | 0.9606 | 0.8389 | 0.9289 | 0.3089 |
| R3 | 10 | 0.9852 | 0.9874 | 0.7250 | 0.9811 | 0.3278 |

The discovery labels have 100% assignment coverage. Raw tables are in `clustering_results.csv` and `incremental_results.csv`; round checkpoints and raw-IQ replay memories are saved after every round.
