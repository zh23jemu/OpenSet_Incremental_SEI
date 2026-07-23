# ADS-B Phase 2 — Classical Feature Development Report

- Calibration data: the same stored-order 10% base-class validation block.
- Classes: 90.
- Samples: 2052.
- Reserved base 30% loaded: no.
- Novel test files opened: no.
- No payload decoding, ICAO address, or ground-truth label enters the feature extractor.

| Feature group | Dimension | Purity k=5 | Purity k=10 | Purity k=20 | Mean |
|---|---:|---:|---:|---:|---:|
| Generic all24 | 24 | 0.2384 | 0.1930 | 0.1549 | 0.1954 |
| ADS-B oscillator | 8 | 0.2229 | 0.1893 | 0.1519 | 0.1881 |
| ADS-B hardware | 23 | 0.2979 | 0.2409 | 0.1852 | 0.2413 |
| ADS-B all | 37 | 0.2948 | 0.2344 | 0.1820 | 0.2371 |

The selected ADS-B hardware feature consists of oscillator/phase-instability,
I/Q non-circularity/DC leakage, and normalized spectral-shape statistics. It
excludes pulse-content features because adding transient/pulse descriptors
slightly reduces mean purity and may increase payload-content sensitivity.

Relative k=10 purity gain over generic all24:

`(0.2409 - 0.1930) / 0.1930 = 24.8%`.

Conclusion: the ADS-B-specific feature is meaningfully better than the generic
24-D view, but purity remains far below the 0.80 global reliability threshold.
It should be used only through CF-LCG local reliability weighting; the global
RF gate should remain closed unless a future validation-only design exceeds
the pre-registered threshold.
