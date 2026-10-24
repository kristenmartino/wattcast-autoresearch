# WattCast benchmark — EIA-930 demand

Retrospective evaluation on real public demand data. This replaces the old synthetic-data performance claims; the two protocols are not comparable.

## Final test: July–December 2025

Per-region XGBoost models are refitted once on January 2024–June 2025 with the same configuration used in validation. Each weekly origin predicts the next 168 hours recursively. The seed contains only the preceding 168 observed hours; subsequent lag and rolling features use predictions. The comparison predicts the value from the same hour one week earlier.

An origin labels the first predicted UTC hour-ending timestamp. The last seed timestamp is origin minus one hour, and lead 1 labels the origin itself. Reporting delay is not modeled.

| Forecast prefix | Scored hours | XGBoost MAPE | Weekly baseline MAPE | XGBoost WAPE | Weekly baseline WAPE | XGBoost bias |
|---|---:|---:|---:|---:|---:|---:|
| 1–24h | 4,776 | 3.73% | 8.85% | 3.45% | 8.45% | -0.18% |
| 1–48h | 9,552 | 4.73% | 8.72% | 4.45% | 8.15% | -0.02% |
| 1–72h | 14,328 | 5.58% | 8.62% | 5.19% | 7.87% | -0.06% |
| 1–168h | 33,432 | 6.84% | 8.58% | 6.53% | 8.00% | -0.82% |

The shorter prefixes use the same eligible seven-day windows; they are not separate daily-origin experiments. Pooled MAPE weights every scored hour equally. WAPE weights errors by demand volume. Bias is signed aggregate error divided by actual demand (positive means overprediction).

## Per-region 168-hour test results

| Region | Origins scored / planned | XGBoost MAPE | Weekly baseline MAPE | XGBoost WAPE | Weekly baseline WAPE | XGBoost bias |
|---|---:|---:|---:|---:|---:|---:|
| CAISO | 24 / 26 | 5.84% | 6.51% | 6.00% | 6.77% | +0.68% |
| ERCOT | 24 / 26 | 5.86% | 5.99% | 5.77% | 5.80% | -1.07% |
| ISONE | 25 / 26 | 9.24% | 12.32% | 9.15% | 12.43% | +0.94% |
| MISO | 24 / 26 | 5.64% | 6.81% | 5.74% | 6.92% | -0.82% |
| NYISO | 26 / 26 | 7.06% | 9.60% | 7.18% | 9.85% | -0.28% |
| PJM | 24 / 26 | 6.59% | 8.56% | 6.64% | 8.73% | -0.46% |
| SECO | 26 / 26 | 7.18% | 9.86% | 7.09% | 9.65% | +0.19% |
| SPP | 26 / 26 | 7.17% | 8.62% | 7.64% | 8.68% | -4.20% |

Equal-weight mean of regional MAPEs: **6.82% XGBoost; 8.53% baseline**. This prevents larger regions or differing eligible-window counts from hiding the regional results.

## Validation and exclusions

Validation uses January–June 2025 with models fitted only on 2024. There was one fixed model configuration and no early stopping or hyperparameter search. The run plan was committed before final-test scoring. No model settings were changed in response to the final test.

| Stage | XGBoost 168h MAPE | Baseline 168h MAPE | Scored / planned region-weeks | Exclusions |
|---|---:|---:|---:|---|
| validation | 6.82% | 9.35% | 194 / 200 | missing_seed: 3, missing_target: 3 |
| test | 6.84% | 8.58% | 199 / 208 | missing_seed: 5, missing_target: 4 |

An origin is excluded if any of its 168 seed hours or 168 target hours is missing, nonfinite, or nonpositive. Gaps are preserved on an hourly UTC grid; targets are never interpolated. The raw quality counts are in the provenance file, and every candidate origin and exclusion reason is in the result JSON. The last 144 hours of validation and last 48 hours of test are outside the full weekly-window schedule.

## Data provenance

Publisher: **U.S. Energy Information Administration**. Source field: **`Demand (MW)`**; forecast, imputed, and adjusted columns are excluded. Timestamps retain **UTC time at end of hour**. Region aliases map to BA codes in [provenance.json](data/provenance.json).

The normalized extract contains **140,352 region-hours** for 2024-01-01 inclusive through 2026-01-01 exclusive. It is committed as [demand.csv.gz](data/demand.csv.gz), SHA-256 `be92c936da78c15da761eb8106d8908e7fc5bc759b8c74aa05df475aa2394f44`. No weather data or synthetic fallback is used.

| Official source | Retrieved (UTC) | Bytes | SHA-256 |
|---|---|---:|---|
| [EIA930_BALANCE_2024_Jan_Jun.csv](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jan_Jun.csv) | 2026-10-24T07:34:47.285800+00:00 | 41,742,812 | `26768c495c3ba987e29d79df2e40d4e741f9dbcff086f53c12ff774465756a10` |
| [EIA930_BALANCE_2024_Jul_Dec.csv](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jul_Dec.csv) | 2026-10-24T07:34:49.923763+00:00 | 47,928,993 | `a602a8e577cfdd2fa3083413eaa4dd8ba7f0d2b26a2986f8f6d22917465ea49f` |
| [EIA930_BALANCE_2025_Jan_Jun.csv](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2025_Jan_Jun.csv) | 2026-10-24T07:34:50.947093+00:00 | 47,565,054 | `fac4bb991dfa1dab3b4b35300ccab54ac99be13a40ce91954db8ddafb2816242` |
| [EIA930_BALANCE_2025_Jul_Dec.csv](https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2025_Jul_Dec.csv) | 2026-10-24T07:34:53.522496+00:00 | 48,256,021 | `1146158b972439db7c7f4cc7d0491e07644ddbf244d03b8ecb44cb2c833d10f1` |

These are the historical files as retrieved in October 2026. They may contain revisions made after the forecast dates. The benchmark does not reproduce publication-time vintages or simulate reporting delay. Treat it as retrospective forecasting evidence, not a live operational performance claim.

## Reproduce and inspect

Install `requirements-lock.txt`, run `pytest -q`, then follow the commands in the [README](../README.md). The committed extract allows offline reproduction; re-downloading EIA data may produce a different vintage. Config, code, data, runtime versions, predictions, and metric hashes are recorded.

- [Fixed configuration](../benchmark_config.json)
- [Frozen pre-test plan](run-plan.json)
- [Validation metrics and coverage](results/validation.json) · [predictions](results/validation.csv.gz)
- [Test metrics and coverage](results/test.json) · [predictions](results/test.csv.gz)
- [Data provenance and quality counts](data/provenance.json)

Final scoring code revision: `88393246d21fbc2ea050451aa463960283c19333`. Run time: 14.0s on `macOS-26.3.1-arm64-arm-64bit`, Python 3.12.13. Runtime versions: numpy 2.5.4, pandas 3.1.0, xgboost 3.4.1, scikit-learn 1.9.1.

Only one chronological test period and eight balancing authorities were studied. Weekly windows are temporally related; no statistical-significance or universal model-superiority claim is made. This demand-only model omits weather, holidays, and other operational inputs.
