"""Fairness metrics over time, drift detection, and root-cause analysis.

All of this is plain, testable statistics. The agents call these functions as tools;
they never compute the numbers themselves.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .simulate import FEATURES


# ------------------------------------------------------------------ metrics
def group_metrics(df: pd.DataFrame, group_col: str = "payer") -> pd.DataFrame:
    """Per month and group: how well the model treats that group.

    tpr               share of patients who were readmitted that the model flagged (equal opportunity)
    selection_rate    share of patients flagged
    calibration_gap   observed readmission rate minus average predicted risk (> 0 means under-predicted)
    """
    def summarize(g):
        pos = g[g["readmitted"] == 1]
        return pd.Series({
            "n": len(g),
            "positives": len(pos),
            "tpr": pos["flagged"].mean() if len(pos) else np.nan,
            "selection_rate": g["flagged"].mean(),
            "observed_rate": g["readmitted"].mean(),
            "predicted_rate": g["score"].mean(),
        })
    out = df.groupby(["month", group_col]).apply(summarize, include_groups=False).reset_index()
    out["calibration_gap"] = out["observed_rate"] - out["predicted_rate"]
    return out.rename(columns={group_col: "group"})


# ------------------------------------------------------------------ detection
@dataclass
class Alert:
    group: str
    metric: str
    first_month: int
    baseline: float
    current: float
    change: float
    p_value: float

    def to_dict(self):
        return asdict(self)


def _two_prop_p(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 1.0 if se == 0 else 2 * stats.norm.sf(abs(x1 / n1 - x2 / n2) / se)


def detect_drift(df: pd.DataFrame, baseline_months: int = 3, alpha: float = 0.01,
                 min_tpr_drop: float = 0.10, min_calibration_shift: float = 0.03,
                 window: int = 2, group_col: str = "payer") -> list[Alert]:
    """Find groups whose fairness got significantly worse after the baseline period.

    Each test pools `window` consecutive months (more power, and a one-month blip cannot
    trigger it). A change counts only if it is statistically significant after a Bonferroni
    correction for every group, window, and metric tested, and large enough to matter.
    """
    alerts = []
    groups = sorted(df[group_col].unique())
    months = sorted(m for m in df["month"].unique() if m >= baseline_months)
    starts = months[: len(months) - window + 1]
    cutoff = alpha / (len(groups) * len(starts) * 2)
    base = df[df["month"] < baseline_months]

    for g in groups:
        b = base[base[group_col] == g]
        b_pos = b[b["readmitted"] == 1]
        b_tpr = b_pos["flagged"].mean()
        b_cal = b["readmitted"].mean() - b["score"].mean()
        found = {}
        for m in starts:
            cur = df[df["month"].between(m, m + window - 1) & (df[group_col] == g)]
            pos = cur[cur["readmitted"] == 1]
            # TPR: did the model start missing more of this group's high-risk patients?
            tpr = pos["flagged"].mean()
            p = _two_prop_p(pos["flagged"].sum(), len(pos), b_pos["flagged"].sum(), len(b_pos))
            if "tpr" not in found and b_tpr - tpr >= min_tpr_drop and p < cutoff:
                found["tpr"] = (m, b_tpr, tpr, p)
            # Calibration: did real risk move above predicted risk?
            cal = cur["readmitted"].mean() - cur["score"].mean()
            se = np.sqrt(cur["readmitted"].var(ddof=1) / len(cur) + b["readmitted"].var(ddof=1) / len(b))
            p_cal = 2 * stats.norm.sf(abs(cal - b_cal) / se) if se > 0 else 1.0
            if "calibration_gap" not in found and cal - b_cal >= min_calibration_shift and p_cal < cutoff:
                found["calibration_gap"] = (m, b_cal, cal, p_cal)
        for metric, (m, baseline, value, p) in found.items():
            # The window may start one month before the change. Pin the start to the first month
            # in the window that crosses the effect size on its own.
            for month in range(m, m + window):
                cur = df[(df["month"] == month) & (df[group_col] == g)]
                if metric == "tpr":
                    single = baseline - cur.loc[cur["readmitted"] == 1, "flagged"].mean()
                    limit = min_tpr_drop
                else:
                    single = (cur["readmitted"].mean() - cur["score"].mean()) - baseline
                    limit = min_calibration_shift
                if single >= limit:
                    m = month
                    break
            alerts.append(Alert(g, metric, int(m), round(float(baseline), 4), round(float(value), 4),
                                round(float(value - baseline), 4), float(p)))
    return sorted(alerts, key=lambda a: (a.first_month, a.group))


# ------------------------------------------------------------------ root cause
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index: how much a feature's distribution moved. > 0.2 is a big shift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        edges = np.unique(np.concatenate([edges, [expected.min() - 1, expected.max() + 1]]))
    e = np.histogram(expected, edges)[0] / len(expected) + 1e-6
    a = np.histogram(np.clip(actual, edges[0], edges[-1]), edges)[0] / len(actual) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def root_cause(df: pd.DataFrame, group: str, start_month: int, baseline_months: int = 3,
               group_col: str = "payer") -> list[dict]:
    """Which features changed for this group, beyond any change in the other groups?

    Uses a difference-in-differences view: shift in the group minus shift in everyone else.
    That separates a group-specific cause from a change that hit every patient.
    """
    before = df[df["month"] < baseline_months]
    after = df[df["month"] >= start_month]
    rows = []
    for f in FEATURES:
        g_shift = psi(before.loc[before[group_col] == group, f].to_numpy(),
                      after.loc[after[group_col] == group, f].to_numpy())
        o_shift = psi(before.loc[before[group_col] != group, f].to_numpy(),
                      after.loc[after[group_col] != group, f].to_numpy())
        g_mean = (after.loc[after[group_col] == group, f].mean()
                  - before.loc[before[group_col] == group, f].mean())
        rows.append({"feature": f, "group_psi": round(g_shift, 4), "others_psi": round(o_shift, 4),
                     "group_specific_shift": round(g_shift - o_shift, 4),
                     "mean_change": round(float(g_mean), 3)})
    def rate(frame, mine):
        sel = frame[group_col] == group
        return frame.loc[sel if mine else ~sel, "readmitted"].mean()
    group_change = rate(after, True) - rate(before, True)
    g_before = before.loc[before[group_col] == group, "readmitted"]
    g_after = after.loc[after[group_col] == group, "readmitted"]
    outcome_p = _two_prop_p(g_after.sum(), len(g_after), g_before.sum(), len(g_before))
    others_change = rate(after, False) - rate(before, False)
    # Did the real outcome change for this group? If not, the inputs changed but the patients did not.
    rows.append({"feature": "readmitted (outcome)", "group_psi": None, "others_psi": None,
                 "group_specific_shift": None, "mean_change": round(float(group_change), 4),
                 "others_mean_change": round(float(others_change), 4),
                 "group_specific_outcome_change": round(float(group_change - others_change), 4),
                 "outcome_change_p_value": float(f"{outcome_p:.3g}")})
    feats = [r for r in rows if r["group_psi"] is not None]
    feats.sort(key=lambda r: r["group_specific_shift"], reverse=True)
    return feats + [rows[-1]]
