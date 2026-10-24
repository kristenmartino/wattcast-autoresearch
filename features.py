"""Shared causal features for training and recursive inference (UTC hour-ending)."""
from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = (1, 2, 3, 6, 12, 24, 48, 168)
WINDOWS = (6, 12, 24, 48, 168)
CALENDAR = ("hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "doy_sin", "doy_cos", "is_weekend")
FEATURES = list(CALENDAR) + [f"demand_lag_{n}h" for n in LAGS]
FEATURES += [f"demand_rolling_{stat}_{n}h" for n in WINDOWS for stat in ("mean", "std")]
FEATURES += ["momentum_3h", "momentum_24h", "ratio_24h", "ratio_168h"]
LOOKBACK = max(*LAGS, *WINDOWS)


def regularize(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep missing hours on the grid; never close gaps by positional shifting."""
    df = raw[["timestamp", "region", "demand_mw"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    if df["timestamp"].isna().any() or df["region"].isna().any():
        raise ValueError("Missing timestamp or region")
    if (df["timestamp"] != df["timestamp"].dt.floor("h")).any():
        raise ValueError("Demand timestamps must be aligned to UTC hours")
    if df.duplicated(["region", "timestamp"]).any():
        raise ValueError("Duplicate region/hour; resolve at ingestion, not feature time")
    df["demand_mw"] = pd.to_numeric(df["demand_mw"], errors="coerce")
    df.loc[~np.isfinite(df["demand_mw"]) | (df["demand_mw"] <= 0), "demand_mw"] = np.nan
    parts = []
    for region, group in df.groupby("region", sort=True):
        group = group.set_index("timestamp").sort_index()
        grid = pd.date_range(group.index.min(), group.index.max(), freq="h", tz="UTC")
        group = group.reindex(grid).rename_axis("timestamp")
        group["region"] = region
        parts.append(group.reset_index())
    if not parts:
        raise ValueError("No demand observations")
    return pd.concat(parts, ignore_index=True)


def calendar_values(ts: pd.Timestamp) -> list[float]:
    return [
        np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24),
        np.sin(2*np.pi*ts.day_of_week/7), np.cos(2*np.pi*ts.day_of_week/7),
        np.sin(2*np.pi*ts.month/12), np.cos(2*np.pi*ts.month/12),
        np.sin(2*np.pi*ts.day_of_year/365.25), np.cos(2*np.pi*ts.day_of_year/365.25),
        float(ts.day_of_week >= 5),
    ]


def feature_row(history: np.ndarray, ts: pd.Timestamp) -> np.ndarray:
    """No target argument: every demand input precedes the forecast hour."""
    past = np.asarray(history, dtype=float)[-LOOKBACK:]
    if len(past) < LOOKBACK or not np.isfinite(past).all() or (past <= 0).any():
        raise ValueError("A complete positive 168-hour seed is required")
    values = calendar_values(ts) + [past[-n] for n in LAGS]
    for n in WINDOWS:
        values.extend([past[-n:].mean(), past[-n:].std(ddof=0)])
    values.extend([past[-1]-past[-3], past[-1]-past[-24],
                   past[-24]/max(past[-24:].mean(), 1),
                   past[-168]/max(past[-168:].mean(), 1)])
    return np.asarray(values, dtype=np.float32)


def build_feature_store(raw: pd.DataFrame) -> pd.DataFrame:
    """Training equivalent of feature_row; target changes cannot affect X[t]."""
    df = regularize(raw)
    ts = df["timestamp"].dt
    for stem, value, period in [("hour", ts.hour, 24), ("dow", ts.day_of_week, 7),
                                 ("month", ts.month, 12), ("doy", ts.day_of_year, 365.25)]:
        df[f"{stem}_sin"] = np.sin(2*np.pi*value/period)
        df[f"{stem}_cos"] = np.cos(2*np.pi*value/period)
    df["is_weekend"] = (ts.day_of_week >= 5).astype(float)
    for n in LAGS:
        df[f"demand_lag_{n}h"] = df.groupby("region")["demand_mw"].shift(n)
    for n in WINDOWS:
        grouped = df.groupby("region")["demand_mw"]
        df[f"demand_rolling_mean_{n}h"] = grouped.transform(lambda x: x.shift(1).rolling(n, min_periods=n).mean())
        df[f"demand_rolling_std_{n}h"] = grouped.transform(lambda x: x.shift(1).rolling(n, min_periods=n).std(ddof=0))
    df["momentum_3h"] = df["demand_lag_1h"]-df["demand_lag_3h"]
    df["momentum_24h"] = df["demand_lag_1h"]-df["demand_lag_24h"]
    df["ratio_24h"] = df["demand_lag_24h"]/df["demand_rolling_mean_24h"].clip(lower=1)
    df["ratio_168h"] = df["demand_lag_168h"]/df["demand_rolling_mean_168h"].clip(lower=1)
    return df


def recursive_forecast(model, history: np.ndarray, timestamps: pd.DatetimeIndex) -> np.ndarray:
    """Append predictions, never actual future demand, to the seed."""
    if len(timestamps) and (timestamps != pd.date_range(timestamps[0], periods=len(timestamps), freq="h")).any():
        raise ValueError("Forecast timestamps must be consecutive hours")
    past = list(np.asarray(history, dtype=float)[-LOOKBACK:])
    predictions = []
    for ts in timestamps:
        x = feature_row(np.asarray(past[-LOOKBACK:]), ts)
        prediction = float(model.predict(x.reshape(1, -1))[0])
        if not np.isfinite(prediction):
            raise ValueError("Model returned a non-finite prediction")
        prediction = max(prediction, 1e-6)
        predictions.append(prediction)
        past.append(prediction)
    return np.asarray(predictions)
