# ADS-B Phase 1 — Closed-set Development Report

- Development boundary: stored-order 60% train / 10% validation.
- Reserved base 30% loaded: no.
- Novel test files opened: no.
- Seed: 41.

| Model | Best validation accuracy | Best epoch |
|---|---:|---:|
| Original WiSig-oriented 1-D CNN | 0.5049 | 46 |
| ADS-B multi-scale long-sequence CNN + attention | 0.6540 | 38 |

Absolute improvement: 0.1491.

Conclusion: increasing epochs alone does not solve the ADS-B problem. The
original network overfits the 60% training block and does not transfer well to
the stored-order validation block. A long-sequence, multi-scale backbone is
justified for ADS-B. The new model still reaches nearly 100% training accuracy
while validation peaks at 65.4%, so domain/capture-order shift and overfitting
remain.

The reserved 30% and novel test files must remain excluded until the full
pipeline and all hyperparameters are frozen.
