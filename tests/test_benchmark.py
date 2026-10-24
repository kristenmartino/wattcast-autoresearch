import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import benchmark
import features
import prepare
import train


def demand(n=600, regions=("A",)):
    times = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.concat([pd.DataFrame({"timestamp": times, "region": r,
                        "demand_mw": 1000+np.arange(n)*2+np.sin(np.arange(n))*20+i*5000})
                      for i,r in enumerate(regions)], ignore_index=True)


def test_target_and_future_cannot_change_current_or_earlier_features():
    raw = demand(regions=("A", "B"))
    cutoff = raw.timestamp.iloc[300]
    before = features.build_feature_store(raw)
    poisoned = raw.copy()
    poisoned.loc[poisoned.timestamp >= cutoff, "demand_mw"] *= 20
    after = features.build_feature_store(poisoned)
    mask = before.timestamp <= cutoff
    np.testing.assert_allclose(before.loc[mask, features.FEATURES], after.loc[mask, features.FEATURES], equal_nan=True)
    assert not np.array_equal(before.demand_mw, after.demand_mw), "control must actually change targets"


def test_rolling_window_excludes_target_and_regions_are_isolated():
    raw = demand(regions=("A", "B"))
    built = features.build_feature_store(raw)
    for region in ("A", "B"):
        values = raw[raw.region == region].demand_mw.to_numpy()
        row = built[built.region == region].iloc[200]
        assert row.demand_rolling_mean_6h == pytest.approx(values[194:200].mean())
        assert row.demand_rolling_std_6h == pytest.approx(values[194:200].std(ddof=0))
        assert row.demand_lag_168h == values[32]


def test_missing_hour_is_not_closed_by_positional_shifting():
    raw = demand()
    absent = raw.timestamp.iloc[200]
    built = features.build_feature_store(raw.drop(index=200)).set_index("timestamp")
    assert pd.isna(built.loc[absent, "demand_mw"])
    assert pd.isna(built.loc[absent+pd.Timedelta(hours=24), "demand_lag_24h"])
    assert built.loc[absent+pd.Timedelta(hours=25), "demand_lag_24h"] == raw.demand_mw.iloc[201]
    assert pd.isna(built.loc[absent+pd.Timedelta(hours=1), "demand_rolling_mean_6h"])


def test_batch_and_recursive_feature_definitions_match():
    raw = demand()
    built = features.build_feature_store(raw)
    for i in (168, 201, 400, 599):
        row = features.feature_row(raw.demand_mw.to_numpy()[:i], raw.timestamp.iloc[i])
        np.testing.assert_allclose(row, built.loc[i, features.FEATURES].to_numpy(dtype=float), rtol=1e-6)


class IncrementLastDemand:
    def predict(self, x):
        return x[:, features.FEATURES.index("demand_lag_1h")]+1


def test_multistep_recursion_uses_predictions_not_future_actuals():
    seed = np.arange(100, 268, dtype=float)
    times = pd.date_range("2025-01-01", periods=168, freq="h", tz="UTC")
    prediction = features.recursive_forecast(IncrementLastDemand(), seed, times)
    np.testing.assert_array_equal(prediction, np.arange(268, 436))


def test_seed_must_be_complete_positive_and_timestamps_contiguous():
    times = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    for seed in [np.ones(167), np.r_[np.ones(167), np.nan], np.r_[np.ones(167), 0]]:
        with pytest.raises(ValueError):
            features.recursive_forecast(IncrementLastDemand(), seed, times)
    with pytest.raises(ValueError):
        features.recursive_forecast(IncrementLastDemand(), np.ones(168), times[[0, 2]])


def test_fit_cutoff_blocks_validation_and_test_targets(monkeypatch):
    calls = []
    class Spy:
        def __init__(self, **params):
            assert "early_stopping_rounds" not in params
        def fit(self, x, y, **kwargs):
            assert "eval_set" not in kwargs
            calls.append((x.copy(), y.copy()))
    monkeypatch.setattr(train, "XGBRegressor", Spy)
    raw = demand()
    cutoff = raw.timestamp.iloc[450].isoformat()
    _, counts = train.fit_models(raw, cutoff, {})
    poisoned = raw.copy()
    poisoned.loc[poisoned.timestamp >= pd.Timestamp(cutoff), "demand_mw"] *= 100
    train.fit_models(poisoned, cutoff, {})
    for a,b in zip(calls[0], calls[1]):
        np.testing.assert_array_equal(a, b)
    assert len(calls[0][1]) == counts["A"]["fit_rows"] == 450-168
    assert pd.Timestamp(counts["A"]["last_fit_timestamp"]) < pd.Timestamp(cutoff)


def test_early_stopping_is_rejected():
    with pytest.raises(ValueError, match="early stopping"):
        train.fit_models(demand(), "2025-01-01T00:00:00Z", {"early_stopping_rounds": 10})


def test_evaluation_future_poisoning_changes_scores_but_not_forecasts():
    raw = demand()
    start, end = raw.timestamp.iloc[300], raw.timestamp.iloc[468]
    models = {"A": IncrementLastDemand()}
    before, coverage = benchmark.evaluate(models, raw, start.isoformat(), end.isoformat(), 168, 168)
    changed = raw.copy()
    changed.loc[changed.timestamp >= start, "demand_mw"] *= 20
    after, _ = benchmark.evaluate(models, changed, start.isoformat(), end.isoformat(), 168, 168)
    np.testing.assert_array_equal(before.xgboost_mw, after.xgboost_mw)
    np.testing.assert_array_equal(before.seasonal_naive_mw, after.seasonal_naive_mw)
    np.testing.assert_array_equal(before.seasonal_naive_mw, raw.demand_mw.iloc[132:300])
    assert benchmark.score(before.actual_mw, before.xgboost_mw) != benchmark.score(after.actual_mw, after.xgboost_mw)
    assert coverage[0]["status"] == "scored"


@pytest.mark.parametrize("gap,reason", [(200, "missing_seed"), (450, "missing_target")])
def test_incomplete_origins_are_reported_not_imputed(gap, reason):
    raw = demand(850)
    start, end = raw.timestamp.iloc[300], raw.timestamp.iloc[804]
    raw.loc[gap, "demand_mw"] = np.nan
    result, coverage = benchmark.evaluate({"A": IncrementLastDemand()}, raw, start.isoformat(), end.isoformat(), 168, 168)
    assert coverage[0]["status"] == reason
    assert result.origin.nunique() >= 1
    assert not result.isna().any().any()


def test_metric_values_and_invalid_inputs():
    metrics = benchmark.score([100, 200], [110, 160])
    assert metrics["n"] == 2
    assert metrics["mape_pct"] == pytest.approx(15)
    assert metrics["wape_pct"] == pytest.approx(100*50/300)
    assert metrics["bias_pct"] == pytest.approx(-10)
    assert metrics["mae_mw"] == 25
    for actual, predicted in [([0], [1]), ([np.nan], [1]), ([], []), ([1], [np.inf]), ([1, 2], [1])]:
        with pytest.raises(ValueError):
            benchmark.score(actual, predicted)


def test_config_rejects_overlap_and_bad_dates():
    config = json.loads(Path("benchmark_config.json").read_text())
    benchmark.validate_config(config)
    for replacement in [{"stride_hours": 24}, {"horizon_hours": 169}, {"test_start": config["validation_start"]}]:
        with pytest.raises(ValueError):
            benchmark.validate_config(config | replacement)


def test_test_stage_requires_explicit_opt_in_and_plan(tmp_path):
    with pytest.raises(ValueError, match="Final test requires"):
        benchmark.run("test", Path("benchmark_config.json"), tmp_path, tmp_path/"test.json")


def test_run_plan_detects_code_config_and_data_tampering(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in benchmark.CODE_FILES:
        Path(name).write_text("# frozen\n")
    config = Path("config.json");config.write_text("{}")
    data = Path("data");data.mkdir()
    (data/"provenance.json").write_text("{}")
    (data/"demand.csv.gz").write_bytes(b"frozen data")
    result = Path("validation.json")
    result.write_text(json.dumps({"stage": "validation", "identity": benchmark.identity(config, data)}))
    plan = Path("run-plan.json")
    benchmark.freeze_plan(plan, result, config, data)
    benchmark.verify_test_plan(plan, config, data)
    for item in [Path("features.py"), config, data/"demand.csv.gz", data/"provenance.json"]:
        old = item.read_bytes();item.write_bytes(old+b" ")
        with pytest.raises(ValueError, match="Frozen"):
            benchmark.verify_test_plan(plan, config, data)
        item.write_bytes(old)
    with pytest.raises(FileExistsError):
        benchmark.freeze_plan(plan, result, config, data)


def test_corrupt_source_cache_fails_without_synthetic_fallback(tmp_path):
    name = prepare.FILES[0]
    (tmp_path/name).write_text("corrupt")
    (tmp_path/(name+".json")).write_text(json.dumps({"url": prepare.SOURCE_ROOT+name, "sha256": "bad"}))
    with pytest.raises(ValueError, match="integrity"):
        prepare.download_source(name, tmp_path, False)
    with pytest.raises(ValueError, match="Missing verified source"):
        prepare.download_source(prepare.FILES[1], tmp_path, False)


def test_committed_extract_matches_recorded_provenance():
    config = json.loads(Path("benchmark_config.json").read_text())
    raw, manifest = benchmark.load_data(Path("benchmarks/data"), config)
    assert len(raw) == 140352
    assert len(raw.region.unique()) == 8
    assert len(manifest["sources"]) == 4
    assert all(s["url"].startswith(prepare.SOURCE_ROOT) and len(s["sha256"]) == 64 for s in manifest["sources"])


def test_ingestion_preserves_gaps_and_rejects_conflicting_duplicates(tmp_path):
    rows = []
    for code in prepare.REGIONS.values():
        rows += [{prepare.SOURCE_COLUMNS[0]: code, prepare.SOURCE_COLUMNS[1]: "01/01/2024 12:00:00 AM", prepare.SOURCE_COLUMNS[2]: "1,000"},
                 {prepare.SOURCE_COLUMNS[0]: code, prepare.SOURCE_COLUMNS[1]: "01/01/2024 02:00:00 AM", prepare.SOURCE_COLUMNS[2]: "0"}]
    source = tmp_path/"eia.csv";pd.DataFrame(rows).to_csv(source, index=False)
    normalized, quality = prepare.normalize([source], "2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z")
    assert len(normalized) == 24
    assert normalized.demand_mw.isna().sum() == 16
    assert quality["absent_hours_inserted"] == 8
    assert normalized.demand_mw.dropna().eq(1000).all()
    pd.DataFrame(rows+[rows[0] | {prepare.SOURCE_COLUMNS[2]: "2,000"}]).to_csv(source, index=False)
    with pytest.raises(ValueError, match="Conflicting duplicate"):
        prepare.normalize([source], "2024-01-01T00:00:00Z", "2024-01-01T03:00:00Z")


def test_small_real_model_fits_and_recurses():
    raw = demand(600)
    models, _ = train.fit_models(raw, raw.timestamp.iloc[500].isoformat(),
                                {"n_estimators": 4, "max_depth": 2, "n_jobs": 1, "random_state": 42})
    predictions = features.recursive_forecast(models["A"], raw.demand_mw.to_numpy()[332:500], pd.DatetimeIndex(raw.timestamp.iloc[500:524]))
    assert len(predictions) == 24
    assert np.isfinite(predictions).all()
