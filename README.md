# WattCast Autoresearch

A reproducible XGBoost benchmark on hourly electricity demand from eight U.S.
balancing authorities. The current pipeline uses real EIA-930 demand, causal
features, separate chronological validation/test periods, and recursive forecasts.

## Benchmark result

The July–December 2025 test scored **33,432 forecast hours** across **199 of 208
candidate region-weeks**. At the 168-hour horizon:

| Model | MAPE | WAPE | Signed bias |
|---|---:|---:|---:|
| XGBoost | 6.84% | 6.53% | -0.82% |
| Same hour last week | 8.58% | 8.00% | +1.05% |

See the [full report](benchmarks/RESULTS.md) for regional results, shorter forecast
prefixes, coverage exclusions, source hashes, and run details. These are retrospective
results on the EIA historical files retrieved in October 2026; they do not reproduce
publication-time revisions or reporting delays. No weather inputs are used.

## What changed

- Rolling demand means and standard deviations now shift by one hour before
  aggregation, excluding the value being predicted.
- All regions retain a continuous UTC hourly grid. Missing hours remain missing;
  lag 24 means 24 hours ago, not 24 surviving rows ago.
- Multi-hour forecasts append their own predictions to the lag history.
- Validation fits on 2024 and scores January–June 2025. The final test refits the
  same configuration on January 2024–June 2025 and scores July–December 2025.
- The configuration is fixed at 400 trees per region, without early stopping or
  hyperparameter search. Code, data, configuration, and runtime versions are locked
  in a plan committed before final-test scoring.

The original 1.44% synthetic validation result is archived under `results/` and
is not comparable to this benchmark. The corrected runs live under `benchmarks/`.

An origin is the first predicted UTC hour-ending timestamp; its last observed
seed hour is one hour earlier. Forecast lead 1 therefore labels the origin.

## Reproduce offline

The normalized EIA extract and its provenance are committed. No API key or data
download is needed for tests or exact-snapshot reproduction.

```bash
git clone https://github.com/kristenmartino/wattcast-autoresearch.git
cd wattcast-autoresearch
uv venv --python 3.12.13
uv pip install --python .venv/bin/python -r requirements-lock.txt
.venv/bin/python -m pytest -q

# New output paths preserve the original evidence.
.venv/bin/python train.py --stage validation --output benchmarks/reproductions/validation.json
.venv/bin/python train.py --stage test --include-test --plan benchmarks/run-plan.json --output benchmarks/reproductions/test.json
```

The test gate verifies the frozen code/config/data/runtime identity and that the
plan is committed. It prevents accidental drift; it cannot prevent an operator
reading public test data outside the runner. A repeat with the same fixed inputs
is a reproduction, not a new independent estimate of model quality.

## Refresh data or run a new study

Official [EIA-930 historical balance files](https://www.eia.gov/electricity/930-content/about.html)
provide the source `Demand (MW)` field. The current ingest supports the four
2024/2025 half-year files and records retrieval times, response metadata, raw
SHA-256 hashes, quality counts, and the normalized extract hash. Download failures
stop the run; there is no synthetic fallback.

```bash
# Keep the original snapshot intact. Network access required.
.venv/bin/python prepare.py --download --cache data/raw-refresh --output data/refreshed-snapshot
```

A new dataset or configuration requires a new validation run and a new run plan.
After examining this test period, use a genuinely new holdout period or nested
evaluation for further model-selection claims. See [program.md](program.md).

## Files and evidence

| Path | Purpose |
|---|---|
| `prepare.py` | EIA ingestion, normalization, quality counts, provenance |
| `features.py` | Shared causal batch and recursive feature definitions |
| `train.py` | Cutoff-restricted model fitting |
| `benchmark.py` | Stage separation, recursive evaluation, baseline, metrics, test gate |
| `benchmark_config.json` | Declared periods, horizon, stride, fixed model settings |
| `benchmarks/data/` | Frozen demand extract and source manifest |
| `benchmarks/run-plan.json` | Pre-test identity lock and protocol |
| `benchmarks/results/` | Predictions, metrics, coverage, software and code identity |
| `render_report.py` | Generates `benchmarks/RESULTS.md` from result artifacts |
| `tests/` | Leakage, temporal alignment, recursion, split, provenance, and artifact checks |
| `results/`, `journal.py` | Historical synthetic experiment records and legacy journal tooling |

CI runs the regression tests and audits the saved predictions against source
targets, the weekly baseline, and published metrics. It does not rerun model
selection or make live-operational performance claims.
