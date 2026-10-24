# WattCast experiment protocol

The current benchmark is a fixed-configuration, demand-only study on a committed
EIA-930 snapshot. It replaces the original synthetic-data tuning exercise.

## Feature and split rules

- Preserve UTC hour-ending timestamps and gaps. No target interpolation or
  positional shifting after missing rows are removed.
- Every demand feature for hour t must depend only on hours before t.
- Training and recursive inference must use the same feature definitions.
- A forecast window starts with 168 pre-origin actuals. Once forecasting starts,
  subsequent lags and rolling statistics use predictions, not future actuals.
- Only calendar features and past demand are used. Adding weather requires
  forecasts actually available at the origin or an explicitly labeled oracle-weather
  experiment; historical observed weather is not an operational forecast input.

## Model development

1. Declare the configuration, periods, horizon, stride, and baseline before running.
2. Run tests, including target/future perturbation and training/inference parity.
3. Commit the code, configuration, normalized extract, and provenance.
4. Evaluate development changes only on the validation stage, writing a new output
   path per experiment. Log attempted settings, not just improvements.
5. Choose a configuration using validation evidence. Run it again if the code,
   data, or configuration changed, then freeze a new plan with:

```bash
.venv/bin/python benchmark.py --freeze-plan <new-plan.json> --validation-result <matching-validation.json>
```

The published study used one configuration, no hyperparameter search, and no
early stopping. Final-test models are refitted on train plus validation with the
same settings. `fit_rows` records the actual number of rows used per region.

## Final test

Commit the plan before invoking `--stage test --include-test --plan <plan.json>`.
The runner checks code, config, data, and runtime identities against the plan.
All result paths are append-only. Test results are never used for early stopping,
parameter choice, or feature choice.

The July–December 2025 test has now been scored and published. It is no longer
unseen for later experiments. Keep it for exact reproduction; use a new holdout
period or nested evaluation for subsequent selection claims. Hashes and flags
protect against accidental drift, not operator access to the public data.

## Report every run

Retain source URLs, retrieval dates, raw/extract hashes, model settings, source
code hashes, runtime versions, fit counts, per-origin predictions, and all
exclusion reasons. Compare the same eligible hours against the weekly baseline.
Report per-region and pooled errors, signed bias, and the vintage/reporting-delay
limits. Keep old synthetic results labeled historical; do not mix their scores
into the current leaderboard.
