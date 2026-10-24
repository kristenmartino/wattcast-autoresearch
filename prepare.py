#!/usr/bin/env python3
"""Download and freeze official EIA-930 demand. Never substitute synthetic data."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

REGIONS = {"ERCOT": "ERCO", "PJM": "PJM", "CAISO": "CISO", "MISO": "MISO",
           "SPP": "SWPP", "NYISO": "NYIS", "ISONE": "ISNE", "SECO": "SOCO"}
SOURCE_ROOT = "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/"
FILES = [f"EIA930_BALANCE_{year}_{half}.csv" for year in (2024, 2025) for half in ("Jan_Jun", "Jul_Dec")]
SOURCE_COLUMNS = ["Balancing Authority", "UTC Time at End of Hour", "Demand (MW)"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def download_source(name: str, cache: Path, allow_download: bool) -> tuple[Path, dict]:
    cache.mkdir(parents=True, exist_ok=True)
    target, sidecar = cache/name, cache/(name+".json")
    url = SOURCE_ROOT+name
    if target.exists() and sidecar.exists():
        metadata = json.loads(sidecar.read_text())
        if metadata["url"] != url or metadata["sha256"] != sha256(target):
            raise ValueError(f"Source cache integrity mismatch: {name}")
        return target, metadata
    if not allow_download:
        raise ValueError(f"Missing verified source {name}; run prepare.py --download")
    started = utc_now()
    temporary = target.with_suffix(".csv.part")
    try:
        request = Request(url, headers={"User-Agent": "WattCast reproducible research benchmark"})
        with urlopen(request, timeout=120) as response, temporary.open("wb") as out:
            content_type = response.headers.get("Content-Type", "")
            etag = response.headers.get("ETag")
            last_modified = response.headers.get("Last-Modified")
            while block := response.read(1024*1024):
                out.write(block)
        with temporary.open("r", encoding="utf-8-sig") as check:
            if "Balancing Authority" not in check.readline():
                raise ValueError("Download was not an EIA balance CSV")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {"url": url, "retrieval_started_at_utc": started, "retrieved_at_utc": utc_now(),
                "sha256": sha256(target), "bytes": target.stat().st_size,
                "etag": etag, "last_modified": last_modified, "content_type": content_type}
    sidecar.write_text(json.dumps(metadata, indent=2)+"\n")
    return target, metadata


def normalize(sources: list[Path], start: str, end: str) -> tuple[pd.DataFrame, dict]:
    parts, source_rows = [], 0
    mapping = {v: k for k, v in REGIONS.items()}
    for source in sources:
        for chunk in pd.read_csv(source, usecols=SOURCE_COLUMNS, dtype="string", chunksize=100000):
            source_rows += len(chunk)
            chunk = chunk[chunk["Balancing Authority"].isin(mapping)].copy()
            if chunk.empty:
                continue
            timestamp = pd.to_datetime(chunk["UTC Time at End of Hour"], format="%m/%d/%Y %I:%M:%S %p", utc=True, errors="raise")
            part = pd.DataFrame({"timestamp": timestamp, "region": chunk["Balancing Authority"].map(mapping),
                                 "demand_mw": pd.to_numeric(chunk["Demand (MW)"].str.replace(",", "", regex=False), errors="coerce")})
            parts.append(part[(part.timestamp >= pd.Timestamp(start)) & (part.timestamp < pd.Timestamp(end))])
    if not parts:
        raise ValueError("No matching EIA rows")
    df = pd.concat(parts, ignore_index=True).sort_values(["region", "timestamp"])
    duplicate_count = int(df.duplicated(["region", "timestamp"]).sum())
    conflicts = df.groupby(["region", "timestamp"])["demand_mw"].nunique(dropna=False)
    if (conflicts > 1).any():
        raise ValueError("Conflicting duplicate region/hour observations")
    df = df.drop_duplicates(["region", "timestamp"]).copy()
    values = df.demand_mw.to_numpy(dtype=float, na_value=np.nan)
    invalid = ~np.isfinite(values) | (values <= 0)
    invalid_counts = df.loc[invalid].groupby("region").size().to_dict()
    df["demand_mw"] = np.where(invalid, np.nan, values)
    if set(df.region) != set(REGIONS):
        raise ValueError("Not all eight balancing authorities are present")
    grid = pd.MultiIndex.from_product([sorted(REGIONS), pd.date_range(start, end, freq="h", inclusive="left")], names=["region", "timestamp"])
    observed = len(df)
    df = df.set_index(["region", "timestamp"]).reindex(grid).reset_index()
    quality = {"raw_source_rows_scanned": source_rows, "selected_observed_rows": observed,
               "identical_duplicate_rows_removed": duplicate_count, "absent_hours_inserted": len(df)-observed,
               "invalid_nonpositive_or_nonfinite_rows_by_region": {k: int(v) for k,v in invalid_counts.items()},
               "missing_target_hours_by_region": {k: int(v) for k,v in df.demand_mw.isna().groupby(df.region).sum().items()}}
    return df, quality


def prepare(config: Path, cache: Path, output: Path, download: bool) -> dict:
    settings = json.loads(config.read_text())
    sources, records = [], []
    for name in FILES:
        print(f"Source: {name}", flush=True)
        path, metadata = download_source(name, cache, download)
        sources.append(path); records.append(metadata)
    df, quality = normalize(sources, settings["train_start"], settings["test_end"])
    output.mkdir(parents=True, exist_ok=True)
    csv = df.to_csv(index=False, date_format="%Y-%m-%dT%H:%M:%SZ", float_format="%.6f", lineterminator="\n").encode()
    extract = output/"demand.csv.gz"
    extract.write_bytes(gzip.compress(csv, mtime=0))
    manifest = {"schema_version": 1, "source_type": "real_public_EIA930",
                "publisher": "U.S. Energy Information Administration", "sources": records,
                "target_column": "Demand (MW)", "excluded_columns": "Forecast, Imputed and Adjusted columns are not used",
                "units": "MW", "time_semantics": "UTC time at end of hour; labels retained as hour-ending timestamps",
                "vintage": "Historical files as retrieved; later revisions may be present. Not an as-of-publication replay.",
                "weather": "No weather inputs: neither observed future weather nor forecast archives used",
                "regions": REGIONS, "period_start_inclusive": settings["train_start"],
                "period_end_exclusive": settings["test_end"], "quality": quality,
                "extract": {"path": "demand.csv.gz", "sha256": sha256(extract), "bytes": extract.stat().st_size, "rows": len(df)},
                "created_at_utc": utc_now()}
    (output/"provenance.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"rows": len(df), "extract_sha256": manifest["extract"]["sha256"], "quality": quality}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Allow official EIA downloads; failures are fatal")
    parser.add_argument("--config", type=Path, default=Path("benchmark_config.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("benchmarks/data"))
    args = parser.parse_args()
    prepare(args.config, args.cache, args.output, args.download)
