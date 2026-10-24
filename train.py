#!/usr/bin/env python3
"""Fit per-region XGBoost models using only rows before a declared cutoff."""
from __future__ import annotations

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features import FEATURES, build_feature_store


def fit_models(raw: pd.DataFrame, cutoff: str, params: dict) -> tuple[dict, dict]:
    if "early_stopping_rounds" in params:
        raise ValueError("This fixed-configuration benchmark does not use early stopping")
    history = raw.loc[raw.timestamp < pd.Timestamp(cutoff)].copy()
    featured = build_feature_store(history)
    models, counts = {}, {}
    for region, frame in featured.groupby("region", sort=True):
        valid = frame.dropna(subset=FEATURES+["demand_mw"])
        if len(valid) < 168:
            raise ValueError(f"Insufficient training rows for {region}")
        x = valid[FEATURES].to_numpy(dtype=np.float32)
        y = valid.demand_mw.to_numpy(dtype=float)
        model = XGBRegressor(**params)
        model.fit(x, y, verbose=False)
        models[region] = model
        counts[region] = {"input_hours": len(frame), "fit_rows": len(valid),
                          "excluded_rows": len(frame)-len(valid),
                          "first_fit_timestamp": valid.timestamp.min().isoformat(),
                          "last_fit_timestamp": valid.timestamp.max().isoformat()}
    return models, counts


if __name__ == "__main__":
    from benchmark import main
    main()
