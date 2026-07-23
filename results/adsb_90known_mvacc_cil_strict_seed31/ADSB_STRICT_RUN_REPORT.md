## Material Passport

- Artifact: ADS-B strict MV-ACC-CIL experiment result
- Protocol: `adsb_90known_strict_mvacc_cil_v1`
- Status: COMPLETED
- Verification status: EXECUTED_ONCE_WITH_FROZEN_CONFIGURATION
- Seed: 31
- Run date: 2026-07-23 (Asia/Shanghai)

## Frozen command

```powershell
.\.venv\Scripts\python.exe .\experiments\exp_adsb_mvacc_cil_strict.py `
  --data_root C:\Users\123\Downloads\Dataset\Dataset `
  --save_dir .\results\adsb_90known_mvacc_cil_strict_seed31 `
  --train_closedset --epochs 20 --use_supcon --adaptive_fusion `
  --disable_cil_baselines --disable_visualization --seed 31
```

No test-result-driven retry or hyperparameter change was performed.

## Data protocol

- Base identities: 90.
- Incremental schedule: 90 -> 100 -> 110 -> 120.
- Base source: per-class stored-order 70% development / 30% evaluation; the development pool is split into 60% backbone training / 10% validation.
- Novel source: `X_train_30Class.npy` is used for discovery and incremental training; `X_test_30Class.npy` is used only for evaluation.
- Exact normalized-IQ overlap across every development/evaluation boundary: zero.
- Capture/session metadata are unavailable. Therefore, exact-sample non-overlap is verified, but physical capture/session independence cannot be proven from this processed corpus.

## Results

| Round | Clusters | Count error | Coverage | Purity | NMI | ARI | Overall acc | New acc | Initial-class drop |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R1 | 10 | 0 | 1.000 | 0.8095 | 0.7563 | 0.6704 | 0.3533 | 0.7100 | 0.1803 |
| R2 | 9 | 1 | 1.000 | 0.6895 | 0.7422 | 0.5484 | 0.3392 | 0.6380 | 0.2103 |
| R3 | 6 | 4 | 1.000 | 0.5699 | 0.7346 | 0.4430 | 0.3273 | 0.5130 | 0.1894 |

Initial 90-class held-out accuracy: 0.4759. Fixed base-evaluation accuracy after R1/R2/R3: 0.2956 / 0.2656 / 0.2865.

## Interpretation boundary

The strict first run shows that MV-ACC-CIL executes end to end without direct test leakage, but it does not establish strong ADS-B performance. R1 finds the correct number of novel identities; R2 and R3 under-cluster. The weak 90-class initial model is a major bottleneck. These test results must not be used to tune and rerun on the same test partition if the current partition is to remain a final evaluation set.
