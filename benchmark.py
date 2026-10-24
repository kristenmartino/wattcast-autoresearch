"""Frozen, rolling-origin benchmark. Test access requires a pre-scoring run plan."""
from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from features import FEATURES, LOOKBACK, recursive_forecast
from prepare import sha256, utc_now
from train import fit_models

CODE_FILES = ("features.py", "train.py", "prepare.py", "benchmark.py")


def validate_config(config: dict) -> None:
    bounds = [pd.Timestamp(config[k]) for k in ("train_start", "validation_start", "test_start", "test_end")]
    if any(t.tzinfo is None for t in bounds) or not all(a < b for a,b in zip(bounds, bounds[1:])):
        raise ValueError("Train/validation/test boundaries must be ordered timezone-aware timestamps")
    h, stride = config["horizon_hours"], config["stride_hours"]
    if not isinstance(h, int) or not 1 <= h <= LOOKBACK or not isinstance(stride, int) or stride < h:
        raise ValueError("Use non-overlapping windows with horizon <= 168 hours")
    if "early_stopping_rounds" in config["model_params"]:
        raise ValueError("Early stopping is not part of this fixed-configuration benchmark")


def load_data(directory: Path, config: dict) -> tuple[pd.DataFrame, dict]:
    manifest = json.loads((directory/"provenance.json").read_text())
    extract = directory/manifest["extract"]["path"]
    if sha256(extract) != manifest["extract"]["sha256"]:
        raise ValueError("Demand extract hash mismatch")
    if manifest["source_type"] != "real_public_EIA930":
        raise ValueError("This benchmark requires explicitly identified EIA-930 data")
    if (manifest["period_start_inclusive"] != config["train_start"] or
            manifest["period_end_exclusive"] != config["test_end"]):
        raise ValueError("Data and configuration periods disagree")
    raw = pd.read_csv(extract)
    raw["timestamp"] = pd.to_datetime(raw.timestamp, utc=True)
    if len(raw) != manifest["extract"]["rows"] or raw.duplicated(["region", "timestamp"]).any():
        raise ValueError("Invalid extract row count or duplicate keys")
    if set(raw.region) != set(manifest["regions"]):
        raise ValueError("Extract and manifest region sets disagree")
    expected = pd.date_range(config["train_start"], config["test_end"], freq="h", inclusive="left")
    for _, frame in raw.groupby("region"):
        if not pd.DatetimeIndex(frame.timestamp).equals(expected):
            raise ValueError("The demand extract must preserve every UTC hour, including missing values")
    return raw, manifest


def score(actual, predicted) -> dict:
    y, p = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if y.shape != p.shape or y.size == 0 or not np.isfinite(y).all() or not np.isfinite(p).all() or (y <= 0).any():
        raise ValueError("Metrics require paired, finite observations and positive targets")
    error = p-y
    return {"n": int(y.size), "mape_pct": float(np.mean(np.abs(error)/y)*100),
            "wape_pct": float(np.abs(error).sum()/y.sum()*100),
            "mae_mw": float(np.abs(error).mean()), "rmse_mw": float(np.sqrt(np.mean(error**2))),
            "bias_pct": float(error.sum()/y.sum()*100)}


def evaluate(models: dict, raw: pd.DataFrame, start: str, end: str, horizon: int, stride: int) -> tuple[pd.DataFrame, list]:
    """Each origin only seeds from earlier actuals; the future target is scored later."""
    stop = pd.Timestamp(end)-pd.Timedelta(hours=horizon)
    origins = pd.date_range(start, stop, freq=f"{stride}h")
    rows, coverage = [], []
    for region, frame in raw.groupby("region", sort=True):
        values = frame.set_index("timestamp").demand_mw.sort_index()
        for origin in origins:
            seed_times = pd.date_range(origin-pd.Timedelta(hours=LOOKBACK), periods=LOOKBACK, freq="h")
            times = pd.date_range(origin, periods=horizon, freq="h")
            seed = values.reindex(seed_times).to_numpy(dtype=float)
            actual = values.reindex(times).to_numpy(dtype=float)
            missing_seed = int((~np.isfinite(seed) | (seed <= 0)).sum())
            missing_target = int((~np.isfinite(actual) | (actual <= 0)).sum())
            reason = "missing_seed" if missing_seed else "missing_target" if missing_target else "scored"
            coverage.append({"region": region, "origin": origin.isoformat(), "status": reason,
                             "missing_seed_hours": missing_seed, "missing_target_hours": missing_target})
            if reason != "scored":
                continue
            predicted = recursive_forecast(models[region], seed, times)
            # At h <= 168, all weekly baseline inputs precede the origin too.
            baseline = seed[:horizon]
            for i, (ts, y, p, b) in enumerate(zip(times, actual, predicted, baseline), 1):
                rows.append({"region": region, "origin": origin.isoformat(), "timestamp": ts.isoformat(),
                             "lead_hour": i, "actual_mw": y, "xgboost_mw": p, "seasonal_naive_mw": b})
        print(f"Evaluated {region}: {sum(x['status']=='scored' for x in coverage if x['region']==region)}/{len(origins)} origins", flush=True)
    result = pd.DataFrame(rows)
    if result.empty or set(result.region) != set(models):
        raise ValueError("Every trained region must have at least one scoreable origin")
    return result, coverage


def summarize(predictions: pd.DataFrame, horizon: int) -> dict:
    result = {}
    for h in sorted({min(n, horizon) for n in (24, 48, 72, 168)}):
        sample = predictions[predictions.lead_hour <= h]
        per_region = {}
        for region, part in sample.groupby("region", sort=True):
            per_region[region] = {name: score(part.actual_mw, part[f"{name}_mw"]) for name in ("xgboost", "seasonal_naive")}
            per_region[region]["origins"] = int(part.origin.nunique())
        pooled = {name: score(sample.actual_mw, sample[f"{name}_mw"]) for name in ("xgboost", "seasonal_naive")}
        result[str(h)] = {"pooled": pooled, "per_region": per_region,
                          "macro_region_mape_pct": {name: float(np.mean([v[name]["mape_pct"] for v in per_region.values()])) for name in pooled},
                          "wape_improvement_over_baseline_pp": pooled["seasonal_naive"]["wape_pct"]-pooled["xgboost"]["wape_pct"]}
    return result


def identity(config_path: Path, data_dir: Path) -> dict:
    return {"code_sha256": {name: sha256(Path(name)) for name in CODE_FILES},
            "config_sha256": sha256(config_path),
            "provenance_sha256": sha256(data_dir/"provenance.json"),
            "extract_sha256": sha256(data_dir/"demand.csv.gz"),
            "runtime": {"python": platform.python_version(), **{name: importlib.metadata.version(name)
                        for name in ("numpy", "pandas", "xgboost", "scikit-learn")}}}


def freeze_plan(path: Path, validation: Path, config_path: Path, data_dir: Path) -> dict:
    if path.exists():
        raise FileExistsError("Use a new run-plan path; do not overwrite the original lock")
    result = json.loads(validation.read_text())
    current = identity(config_path, data_dir)
    if result["stage"] != "validation" or result["identity"] != current:
        raise ValueError("Validation result does not match the current frozen inputs")
    plan = {"schema_version": 1, "created_at_utc": utc_now(), "identity": current,
            "validation_result_sha256": sha256(validation),
            "selection": "One fixed configuration declared before validation. No tuning or early stopping.",
            "test_protocol": "Refit the same per-region configuration on train+validation; rolling 168-hour recursive forecasts seeded with pre-origin actuals only."}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2)+"\n")
    return plan


def verify_test_plan(plan_path: Path, config_path: Path, data_dir: Path) -> dict:
    plan = json.loads(plan_path.read_text())
    if plan["identity"] != identity(config_path, data_dir):
        raise ValueError("Frozen code/config/data changed after validation; a new protocol and validation are required")
    return plan


def require_committed_plan(plan_path: Path) -> None:
    """A local commit must contain the exact plan before final scoring starts."""
    root = Path(subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True).stdout.strip())
    relative = plan_path.resolve().relative_to(root).as_posix()
    stored = subprocess.run(["git", "show", f"HEAD:{relative}"], capture_output=True, check=True).stdout
    if stored != plan_path.read_bytes():
        raise ValueError("Commit the frozen run plan before accessing the final test")


def run(stage: str, config_path: Path, data_dir: Path, output: Path, include_test: bool = False, plan_path: Path | None = None) -> dict:
    if output.exists() or output.with_suffix(".csv.gz").exists():
        raise FileExistsError("Result artifacts are append-only; use a new output path to reproduce a run")
    config = json.loads(config_path.read_text())
    validate_config(config)
    if stage == "test":
        if not include_test or plan_path is None:
            raise ValueError("Final test requires --include-test and a frozen --plan")
        verify_test_plan(plan_path, config_path, data_dir)
        require_committed_plan(plan_path)
    raw, provenance = load_data(data_dir, config)
    if stage == "validation":
        start, end = config["validation_start"], config["test_start"]
    elif stage == "test":
        start, end = config["test_start"], config["test_end"]
    else:
        raise ValueError("Unknown stage")
    began = time.perf_counter()
    print(f"Fit cutoff (exclusive): {start}", flush=True)
    # Validation fits on 2024. Final test refits on 2024 + Jan-Jun 2025.
    models, counts = fit_models(raw, start, config["model_params"])
    predictions, coverage = evaluate(models, raw, start, end, config["horizon_hours"], config["stride_hours"])
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_path = output.with_suffix(".csv.gz")
    prediction_path.write_bytes(gzip.compress(predictions.to_csv(index=False, float_format="%.9f", lineterminator="\n").encode(), mtime=0))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    result = {"schema_version": 1, "stage": stage, "created_at_utc": utc_now(), "git_commit": revision,
              "identity": identity(config_path, data_dir), "config": config, "features": FEATURES,
              "dataset_source_type": provenance["source_type"], "fit_rows": counts,
              "evaluation_period": {"start_inclusive": start, "end_exclusive": end},
              "metrics": summarize(predictions, config["horizon_hours"]), "coverage": coverage,
              "prediction_artifact": {"path": prediction_path.name, "sha256": sha256(prediction_path), "rows": len(predictions)},
              "run_plan_sha256": sha256(plan_path) if plan_path is not None else None,
              "software": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "xgboost", "scikit-learn")},
              "python": platform.python_version(), "platform": platform.platform(),
              "elapsed_seconds": time.perf_counter()-began,
              "limits": ["Retrospective revised demand, not publication-vintage data; no reporting latency simulated.",
                         "Calendar and past demand only; no weather inputs.",
                         "Weekly origins, fixed models within each stage; earlier observed test weeks may seed later origins.",
                         "Incomplete seed/target windows excluded and enumerated; no target interpolation or outlier trimming.",
                         "Historical synthetic/leaky scores are not comparable to this real-data protocol."]}
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps(result["metrics"][str(config["horizon_hours"])]["pooled"], indent=2), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("validation", "test"), default="validation")
    parser.add_argument("--config", type=Path, default=Path("benchmark_config.json"))
    parser.add_argument("--data", type=Path, default=Path("benchmarks/data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--freeze-plan", type=Path)
    parser.add_argument("--validation-result", type=Path, default=Path("benchmarks/results/validation.json"))
    args = parser.parse_args()
    if args.freeze_plan:
        freeze_plan(args.freeze_plan, args.validation_result, args.config, args.data)
    else:
        run(args.stage, args.config, args.data, args.output or Path(f"benchmarks/results/{args.stage}.json"), args.include_test, args.plan)


if __name__ == "__main__":
    main()
