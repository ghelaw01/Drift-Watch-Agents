"""Make one batch of simulated prediction logs and land it as a Parquet file (bronze input)."""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone

import pandas as pd

from driftwatch.simulate import generate

from . import scenarios
from .config import LOG_COLUMNS


def resolve_seed(seed: str | int) -> int:
    """'auto' (or empty) gives a new seed each run; anything else must be an integer."""
    if seed in (None, "", "auto"):
        return int(time.time()) % 1_000_000
    return int(seed)


def make_batch_id(scenario: str, run_id: str | None = None, now: datetime | None = None) -> str:
    """A readable, file-name-safe batch id. Uses the job run id when there is one."""
    now = now or datetime.now(timezone.utc)
    tail = run_id if run_id not in (None, "", "local") else now.strftime("%Y%m%dT%H%M%S")
    return re.sub(r"[^A-Za-z0-9_\-]", "_", f"{scenario}-{tail}")


def make_batch(scenario: str, seed: int, batch_id: str, months: int = 12,
               patients_per_month: int = 4000, now: datetime | None = None) -> pd.DataFrame:
    """Call the core simulator and add the batch columns. Column order and types are fixed."""
    df = generate(scenarios.build(scenario, seed, months, patients_per_month))
    df.insert(0, "batch_id", batch_id)
    df.insert(1, "scenario", scenario)
    # UTC-aware, so Parquet marks it as UTC and Spark reads it as TIMESTAMP (not TIMESTAMP_NTZ).
    df.insert(2, "generated_at", pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("UTC"))
    casts = {"int": "int32", "double": "float64", "string": "object"}
    for col, kind in LOG_COLUMNS.items():
        if kind in casts:
            df[col] = df[col].astype(casts[kind])
    return df[list(LOG_COLUMNS)]


def write_batch(df: pd.DataFrame, landing_dir: str) -> str:
    """Write the batch as one Parquet file. On Databricks, landing_dir is a UC Volume path."""
    os.makedirs(landing_dir, exist_ok=True)
    batch_id = str(df["batch_id"].iloc[0])
    path = os.path.join(landing_dir, f"{batch_id}.parquet")
    # Spark cannot read nanosecond timestamps by default, so store microseconds.
    df.to_parquet(path, index=False, engine="pyarrow", coerce_timestamps="us", allow_truncated_timestamps=True)
    return path
