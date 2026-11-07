"""Gold-layer fairness tables, in pandas.

`gold_metrics` is the per-batch, per-month, per-group version of
driftwatch.analysis.group_metrics. `watchlist` compares each month with the baseline
months and marks large changes.

The watchlist is only an effect-size screen. The real alerts (with significance tests and
a Bonferroni correction) come from driftwatch.analysis.detect_drift in the audit job.

spark_gold.py has the same logic in PySpark for the pipeline; a Spark test checks both match.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import BASELINE_MONTHS, MIN_CALIBRATION_SHIFT, MIN_TPR_DROP

METRIC_COLUMNS = ["batch_id", "scenario", "month", "group", "n", "positives", "tpr", "selection_rate",
                  "observed_rate", "predicted_rate", "calibration_gap"]
WATCH_COLUMNS = ["batch_id", "scenario", "month", "group", "n", "tpr", "baseline_tpr", "tpr_change",
                 "calibration_gap", "baseline_calibration_gap", "calibration_change", "tpr_drop_flag",
                 "calibration_flag", "in_baseline"]


def gold_metrics(silver: pd.DataFrame, group_col: str = "payer") -> pd.DataFrame:
    """Per batch, month, and group: n, positives, TPR, selection rate, calibration gap."""
    from driftwatch.analysis import group_metrics

    frames = []
    for (batch_id, scenario), part in silver.groupby(["batch_id", "scenario"], sort=True):
        m = group_metrics(part, group_col)
        m.insert(0, "batch_id", batch_id)
        m.insert(1, "scenario", scenario)
        frames.append(m)
    if not frames:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    out = pd.concat(frames, ignore_index=True)
    out["month"] = out["month"].astype(int)
    out[["n", "positives"]] = out[["n", "positives"]].astype(int)
    return out[METRIC_COLUMNS].sort_values(["batch_id", "group", "month"]).reset_index(drop=True)


def _baseline(silver: pd.DataFrame, baseline_months: int, group_col: str) -> pd.DataFrame:
    base = silver[silver["month"] < baseline_months]
    rows = []
    for (batch_id, group), g in base.groupby(["batch_id", group_col]):
        pos = g[g["readmitted"] == 1]
        rows.append({"batch_id": batch_id, "group": group,
                     "baseline_tpr": pos["flagged"].mean() if len(pos) else np.nan,
                     "baseline_calibration_gap": g["readmitted"].mean() - g["score"].mean()})
    return pd.DataFrame(rows, columns=["batch_id", "group", "baseline_tpr", "baseline_calibration_gap"])


def watchlist(silver: pd.DataFrame, baseline_months: int = BASELINE_MONTHS, group_col: str = "payer",
              min_tpr_drop: float = MIN_TPR_DROP,
              min_calibration_shift: float = MIN_CALIBRATION_SHIFT) -> pd.DataFrame:
    """Each month compared with the pooled baseline months of the same batch and group."""
    m = gold_metrics(silver, group_col)
    if m.empty:
        return pd.DataFrame(columns=WATCH_COLUMNS)
    out = m.merge(_baseline(silver, baseline_months, group_col), on=["batch_id", "group"], how="left")
    out["tpr_change"] = out["tpr"] - out["baseline_tpr"]
    out["calibration_change"] = out["calibration_gap"] - out["baseline_calibration_gap"]
    out["in_baseline"] = out["month"] < baseline_months
    # Small tolerance so float rounding gives the same flags in pandas and Spark.
    eps = 1e-9
    out["tpr_drop_flag"] = (~out["in_baseline"]) & (-out["tpr_change"] >= min_tpr_drop - eps)
    out["calibration_flag"] = (~out["in_baseline"]) & (out["calibration_change"] >= min_calibration_shift - eps)
    return out[WATCH_COLUMNS].sort_values(["batch_id", "group", "month"]).reset_index(drop=True)
