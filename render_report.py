"""Render reported metrics from committed JSON evidence; no model calls."""
import json
from collections import Counter
from pathlib import Path


def main():
    root = Path("benchmarks")
    provenance = json.loads((root/"data/provenance.json").read_text())
    test = json.loads((root/"results/test.json").read_text())
    validation = json.loads((root/"results/validation.json").read_text())
    lines = ["# WattCast benchmark — EIA-930 demand", "",
             "Retrospective evaluation on real public demand data. This replaces the old synthetic-data performance claims; the two protocols are not comparable.", "",
             "## Final test: July–December 2025", "",
             "Per-region XGBoost models are refitted once on January 2024–June 2025 with the same configuration used in validation. Each weekly origin predicts the next 168 hours recursively. The seed contains only the preceding 168 observed hours; subsequent lag and rolling features use predictions. The comparison predicts the value from the same hour one week earlier.", "",
             "An origin labels the first predicted UTC hour-ending timestamp. The last seed timestamp is origin minus one hour, and lead 1 labels the origin itself. Reporting delay is not modeled.", "",
             "| Forecast prefix | Scored hours | XGBoost MAPE | Weekly baseline MAPE | XGBoost WAPE | Weekly baseline WAPE | XGBoost bias |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for h, summary in test["metrics"].items():
        x,b = summary["pooled"]["xgboost"], summary["pooled"]["seasonal_naive"]
        lines.append(f"| 1–{h}h | {x['n']:,} | {x['mape_pct']:.2f}% | {b['mape_pct']:.2f}% | {x['wape_pct']:.2f}% | {b['wape_pct']:.2f}% | {x['bias_pct']:+.2f}% |")
    summary = test["metrics"]["168"]
    lines += ["", "The shorter prefixes use the same eligible seven-day windows; they are not separate daily-origin experiments. Pooled MAPE weights every scored hour equally. WAPE weights errors by demand volume. Bias is signed aggregate error divided by actual demand (positive means overprediction).", "",
              "## Per-region 168-hour test results", "",
              "| Region | Origins scored / planned | XGBoost MAPE | Weekly baseline MAPE | XGBoost WAPE | Weekly baseline WAPE | XGBoost bias |", "|---|---:|---:|---:|---:|---:|---:|"]
    for region, metrics in summary["per_region"].items():
        x,b = metrics["xgboost"], metrics["seasonal_naive"]
        planned = sum(c["region"] == region for c in test["coverage"])
        lines.append(f"| {region} | {metrics['origins']} / {planned} | {x['mape_pct']:.2f}% | {b['mape_pct']:.2f}% | {x['wape_pct']:.2f}% | {b['wape_pct']:.2f}% | {x['bias_pct']:+.2f}% |")
    macro = summary["macro_region_mape_pct"]
    lines += ["", f"Equal-weight mean of regional MAPEs: **{macro['xgboost']:.2f}% XGBoost; {macro['seasonal_naive']:.2f}% baseline**. This prevents larger regions or differing eligible-window counts from hiding the regional results.", "",
              "## Validation and exclusions", "",
              "Validation uses January–June 2025 with models fitted only on 2024. There was one fixed model configuration and no early stopping or hyperparameter search. The run plan was committed before final-test scoring. No model settings were changed in response to the final test.", "",
              "| Stage | XGBoost 168h MAPE | Baseline 168h MAPE | Scored / planned region-weeks | Exclusions |", "|---|---:|---:|---:|---|"]
    for stage in (validation, test):
        counts = Counter(c["status"] for c in stage["coverage"])
        x = stage["metrics"]["168"]["pooled"]
        detail = ", ".join(f"{k}: {v}" for k,v in sorted(counts.items()) if k != "scored") or "none"
        lines.append(f"| {stage['stage']} | {x['xgboost']['mape_pct']:.2f}% | {x['seasonal_naive']['mape_pct']:.2f}% | {counts['scored']} / {len(stage['coverage'])} | {detail} |")
    lines += ["", "An origin is excluded if any of its 168 seed hours or 168 target hours is missing, nonfinite, or nonpositive. Gaps are preserved on an hourly UTC grid; targets are never interpolated. The raw quality counts are in the provenance file, and every candidate origin and exclusion reason is in the result JSON. The last 144 hours of validation and last 48 hours of test are outside the full weekly-window schedule.", "",
              "## Data provenance", "",
              f"Publisher: **{provenance['publisher']}**. Source field: **`Demand (MW)`**; forecast, imputed, and adjusted columns are excluded. Timestamps retain **UTC time at end of hour**. Region aliases map to BA codes in [provenance.json](data/provenance.json).", "",
              f"The normalized extract contains **{provenance['extract']['rows']:,} region-hours** for 2024-01-01 inclusive through 2026-01-01 exclusive. It is committed as [demand.csv.gz](data/demand.csv.gz), SHA-256 `{provenance['extract']['sha256']}`. No weather data or synthetic fallback is used.", "",
              "| Official source | Retrieved (UTC) | Bytes | SHA-256 |", "|---|---|---:|---|"]
    for s in provenance["sources"]:
        name = s["url"].rsplit("/",1)[-1]
        lines.append(f"| [{name}]({s['url']}) | {s['retrieved_at_utc']} | {s['bytes']:,} | `{s['sha256']}` |")
    lines += ["", "These are the historical files as retrieved in October 2026. They may contain revisions made after the forecast dates. The benchmark does not reproduce publication-time vintages or simulate reporting delay. Treat it as retrospective forecasting evidence, not a live operational performance claim.", "",
              "## Reproduce and inspect", "",
              "Install `requirements-lock.txt`, run `pytest -q`, then follow the commands in the [README](../README.md). The committed extract allows offline reproduction; re-downloading EIA data may produce a different vintage. Config, code, data, runtime versions, predictions, and metric hashes are recorded.", "",
              "- [Fixed configuration](../benchmark_config.json)",
              "- [Frozen pre-test plan](run-plan.json)",
              "- [Validation metrics and coverage](results/validation.json) · [predictions](results/validation.csv.gz)",
              "- [Test metrics and coverage](results/test.json) · [predictions](results/test.csv.gz)",
              "- [Data provenance and quality counts](data/provenance.json)", "",
              f"Final scoring code revision: `{test['git_commit']}`. Run time: {test['elapsed_seconds']:.1f}s on `{test['platform']}`, Python {test['python']}. Runtime versions: "+", ".join(f"{k} {v}" for k,v in test["software"].items())+".", "",
              "Only one chronological test period and eight balancing authorities were studied. Weekly windows are temporally related; no statistical-significance or universal model-superiority claim is made. This demand-only model omits weather, holidays, and other operational inputs.", ""]
    (root/"RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
