"""Audit published benchmark evidence without rerunning model selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchmark
from prepare import sha256


@pytest.mark.parametrize("stage", ["validation", "test"])
def test_prediction_artifacts_match_data_metrics_and_frozen_code(stage):
    path = Path(f"benchmarks/results/{stage}.json")
    if not path.exists():
        pytest.skip("Run artifact has not been produced yet")
    result = json.loads(path.read_text())
    assert result["config"] == json.loads(Path("benchmark_config.json").read_text())
    raw, manifest = benchmark.load_data(Path("benchmarks/data"), result["config"])
    assert result["dataset_source_type"] == manifest["source_type"] == "real_public_EIA930"
    for name, digest in result["identity"]["code_sha256"].items():
        assert sha256(Path(name)) == digest
    assert result["identity"]["config_sha256"] == sha256(Path("benchmark_config.json"))
    assert result["identity"]["provenance_sha256"] == sha256(Path("benchmarks/data/provenance.json"))
    assert result["identity"]["extract_sha256"] == sha256(Path("benchmarks/data/demand.csv.gz"))
    prediction_file = path.parent/result["prediction_artifact"]["path"]
    assert sha256(prediction_file) == result["prediction_artifact"]["sha256"]
    rows = pd.read_csv(prediction_file)
    assert len(rows) == result["prediction_artifact"]["rows"]
    assert not rows.duplicated(["region", "origin", "lead_hour"]).any()
    for name in ("timestamp", "origin"):
        rows[name] = pd.to_datetime(rows[name], utc=True)
    assert (rows.timestamp == rows.origin+pd.to_timedelta(rows.lead_hour-1, unit="h")).all()
    assert (rows.timestamp >= pd.Timestamp(result["evaluation_period"]["start_inclusive"])).all()
    assert (rows.timestamp < pd.Timestamp(result["evaluation_period"]["end_exclusive"])).all()
    joined = rows.merge(raw, on=["timestamp", "region"], how="left", validate="many_to_one")
    np.testing.assert_allclose(joined.actual_mw, joined.demand_mw, rtol=0, atol=1e-6)
    baseline_source = raw.rename(columns={"timestamp": "baseline_timestamp", "demand_mw": "baseline_actual"})
    rows["baseline_timestamp"] = rows.timestamp-pd.Timedelta(hours=168)
    joined = rows.merge(baseline_source, on=["baseline_timestamp", "region"], how="left", validate="many_to_one")
    assert (joined.baseline_timestamp < joined.origin).all()
    np.testing.assert_allclose(joined.seasonal_naive_mw, joined.baseline_actual, rtol=0, atol=1e-6)
    for horizon, summary in result["metrics"].items():
        part = rows[rows.lead_hour <= int(horizon)]
        for model in ("xgboost", "seasonal_naive"):
            actual_metrics = benchmark.score(part.actual_mw, part[f"{model}_mw"])
            assert summary["pooled"][model] == pytest.approx(actual_metrics, rel=1e-8, abs=1e-8)
            for region, region_summary in summary["per_region"].items():
                regional = part[part.region == region]
                assert region_summary[model] == pytest.approx(benchmark.score(regional.actual_mw, regional[f"{model}_mw"]), rel=1e-8, abs=1e-8)
    scored = [c for c in result["coverage"] if c["status"] == "scored"]
    assert len(rows) == len(scored)*result["config"]["horizon_hours"]
    for region, counts in result["fit_rows"].items():
        cutoff = pd.Timestamp(result["evaluation_period"]["start_inclusive"])
        assert pd.Timestamp(counts["last_fit_timestamp"]) < cutoff
        assert counts["fit_rows"]+counts["excluded_rows"] == counts["input_hours"]
    if stage == "test":
        assert result["run_plan_sha256"] == sha256(Path("benchmarks/run-plan.json"))
        plan = json.loads(Path("benchmarks/run-plan.json").read_text())
        assert plan["identity"] == result["identity"]
        assert plan["validation_result_sha256"] == sha256(Path("benchmarks/results/validation.json"))
        assert pd.Timestamp(plan["created_at_utc"]) < pd.Timestamp(result["created_at_utc"])
