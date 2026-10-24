# WattCast Autoresearch

An experiment in using a coding agent to tune XGBoost on synthetic hourly
electricity-demand data, following [Karpathy's autoresearch pattern](https://github.com/karpathy/autoresearch).

`prepare.py` generates data for eight region labels. Its EIA/Open-Meteo fetch
function is a placeholder and returns no data, including when `EIA_API_KEY` is
set. This repository does not currently benchmark forecasts on real grid data.

## Evaluation limits

The current feature preparation includes the demand being predicted in its
rolling demand statistics. Validation rows also contain observed demand lags;
the evaluator does not recursively feed predictions into subsequent hours.
These features and the evaluation protocol need correction before the scores
can support a forward-forecasting claim.

The agent repeatedly optimizes the same validation set. There is no separate
untouched test set in the supplied workflow, so improvements on that set do
not establish generalization.

The committed [experiment journal](results/JOURNAL.md) and
[changelog](results/changelog.md) record a baseline of **1.7232% MAPE** and a
best score of **1.4411%** for that historical experiment. These are recorded
validation scores, not independently reproduced results or leakage-free forecast
accuracy. The previous README's 3.13% baseline does not match those records and
has been removed.

## Run locally

```bash
git clone https://github.com/kristenmartino/wattcast-autoresearch.git
cd wattcast-autoresearch
pip install -r requirements.txt
python prepare.py    # synthetic data, including when an API key is set
python train.py      # diagnostic validation metrics; see limits above
```

To run an agent-driven experiment, provide a coding agent and follow
[`program.md`](program.md). The repository supplies training scripts and
instructions; the agent performs the edit/run/keep-or-revert loop.

```bash
claude "Read program.md and run a bounded experiment session on train.py"
```

## Files

| File | Purpose |
|---|---|
| `prepare.py` | Synthetic data generation, feature preparation, chronological train/validation split |
| `train.py` | Feature selection, XGBoost training, validation scoring |
| `program.md` | Agent instructions and experiment constraints |
| `journal.py` | Summarizes local experiment history |
| `results/JOURNAL.md` | Committed historical experiment summary |
| `results/changelog.md` | Historical notes on attempted changes |

`train.py` reports pooled row-level MAPE across the validation data, plus
per-region diagnostics. Training uses validation data for early stopping.
Compute time varies with the chosen parameters; there is no enforced runtime
budget.

## Before publishing a forecasting benchmark

1. Implement and verify real-data ingestion with recorded source dates.
2. Remove target leakage and match feature construction to the intended forecast horizon.
3. Reserve an independent test period outside the agent's tuning loop.
4. Compare against a seasonal baseline and report per-region errors, data provenance,
   model settings, and evaluation dates.

Adding a region to the current generator adds another synthetic series. It does
not connect an EIA series. Changing the optimization metric requires updating
the scorer and experiment instructions together.
