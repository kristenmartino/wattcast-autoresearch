# program.md — WattCast Research Directives

> **You are an autonomous ML research agent.** Your job is to make `train.py` achieve 
> the lowest possible MAPE on the validation set. You read this file for strategy, 
> modify `train.py`, run it, check `results/metrics.json`, and decide whether to keep 
> or revert your change. Then repeat.

---

## Objective

Minimize **MAPE (Mean Absolute Percentage Error)** for energy demand forecasting 
on the synthetic validation data for eight region labels. Establish a baseline
for each session and record the data hash and settings. The historical journal
starts at 1.7232%; that score is not real-grid forecasting accuracy.

The current preparation leaks the target through rolling demand statistics, and
the evaluator uses observed validation lags rather than recursive forecasts.
Treat this as a diagnostic tuning exercise until those issues are corrected.
Repeated validation improvements do not establish independent test performance.
See README.md for the evaluation limits.

---

## What You Can Modify

**`train.py` is the ONLY file you may edit.** Specifically:

1. **Feature engineering** (`engineer_features()`) — Create new derived features.
   Add them to the feature list constants.
2. **Feature selection** (the feature list constants) — Add, remove, or reorder 
   features. Try aggressive pruning of noisy features.
3. **Hyperparameters** (`get_model_params()`) — Tune XGBoost parameters. Try 
   different learning rates, depths, regularization strengths.
4. **Preprocessing** (`preprocess()`) — Try log transforms, outlier clipping,
   feature scaling, binning strategies.
5. **Training strategy** (`train_model()`) — Experiment with early stopping 
   thresholds, custom loss functions, or ensemble approaches.

## What You Must NOT Modify

- `prepare.py` — Data preparation is fixed
- The `evaluate()` function in `train.py` — Evaluation must stay consistent
- The `METRIC`, `DATA_DIR`, `RESULTS_DIR`, `TARGET_COL` constants
- The output format of `results/metrics.json`

---

## Research Directions (Priority Order)

### Phase 1: Feature Engineering (experiments 1-30)
Focus on extracting more signal from existing data:
- **Temperature non-linearity**: The demand-temperature relationship is U-shaped 
  (heating + cooling). Try piecewise features, cubic terms, or spline-like 
  approximations with multiple breakpoints.
- **Temporal interactions**: hour × month, hour × is_weekend, temperature × hour.
  Demand patterns shift by season AND by time of day.
- **Demand momentum**: difference between recent lags (lag_1h - lag_3h). Captures 
  whether demand is ramping up or down.
- **Weather change rates**: temperature delta over last 3, 6, 12 hours. Rapid 
  temperature drops spike heating demand.
- **Rolling weather stats**: min/max temperature in last 24h, precipitation 
  accumulation, sustained wind periods.
- **Regional features**: encode region as a feature so one model can learn 
  region-specific patterns.

### Phase 2: Feature Pruning (experiments 30-50)
Remove noise to help the model generalize:
- Check feature importance from Phase 1 results
- Aggressively drop features with near-zero importance
- Test removing correlated feature pairs (keep the stronger one)
- Try L1-heavy regularization to identify dispensable features

### Phase 3: Hyperparameter Optimization (experiments 50-80)
Systematic tuning:
- **Learning rate + n_estimators**: try lower LR (0.01) with more trees (2000+)
- **Tree structure**: vary max_depth (4-12), min_child_weight (1-20)
- **Regularization**: sweep reg_alpha (0-5), reg_lambda (0-10), gamma (0-1)
- **Sampling**: vary subsample (0.5-1.0), colsample_bytree (0.5-1.0)
- **Binning**: try max_bin values (64, 128, 256, 512)

### Phase 4: Advanced Strategies (experiments 80+)
If MAPE is still above 2.5%, try:
- **Per-region models**: train separate models for each region
- **Stacking**: XGBoost + LightGBM ensemble
- **Custom objective**: weighted MAPE that penalizes peak-hour errors more
- **Target transform**: predict log(demand) instead of raw demand
- **Quantile features**: replace rolling mean/std with rolling quantiles (10th, 50th, 90th)
- **Calendar features**: add holiday indicators, daylight savings transitions

---

## Experiment Protocol

For each experiment:
1. Read the recent history in `results/history.jsonl`
2. Choose ONE targeted modification based on what's worked/failed
3. Edit `train.py` — change one thing at a time for clean signal
4. Run `python train.py`
5. Read `results/metrics.json`
6. If MAPE improved → `git commit -am "experiment N: [description] — MAPE X.XX%"`
7. If MAPE worsened → `git checkout -- train.py` (revert)
8. **Every 5 experiments** → Run `python journal.py` to update the research journal
9. **At end of session** → Run `python journal.py` one final time before stopping

**Golden rule: one variable per experiment.** If you change both features AND 
hyperparameters, you won't know which one helped (or hurt).

---

## Documentation Requirements

**This is critical.** The human will read `results/JOURNAL.md` in the morning to 
understand what happened overnight. The journal generator handles most of this 
automatically, but you MUST also maintain `results/changelog.md` with your own 
narrative notes. After every experiment, append a short entry:

```markdown
### Experiment N — [KEEP/DISCARD]
**What I changed:** [one sentence]
**Hypothesis:** [why I thought this would help]
**Result:** MAPE X.XX% (Δ ±X.XX%)
**What I learned:** [one sentence — what does this tell us about the model?]
**Next move:** [what this result suggests trying next]
```

The changelog is your lab notebook. Write it like a researcher documenting their
reasoning, not just logging numbers. The human needs to understand the *why* behind
each experiment — not just the outcome.

Run `python journal.py` every 5 experiments to regenerate the full journal with
trend analysis, pattern detection, and region-level breakdowns.

---

## Session completion

Set an experiment count or time budget before starting. Record each attempted
change, validation score, settings, and keep/discard decision. A session that
finds no improvement is still a useful result. Label all output as synthetic
validation work; do not translate score changes into real-grid accuracy claims.

---

## Notes for the Agent

- This repository uses generated data and has no working EIA fetch or serving cadence.
- Compare region-level errors within the generated dataset; the region names do
  not establish performance on the corresponding real grids.
- Watch for overfitting: if train MAPE drops but val MAPE doesn't, increase regularization.
- The `demand_lag_24h` and `demand_lag_168h` features are usually the most important.
  Build on that signal — e.g., `demand_lag_24h / demand_rolling_mean_168h` captures 
  whether today is abnormal relative to the weekly pattern.
