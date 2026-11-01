"""The audit tools. Agents call these; every number they report comes from here.

The same functions are served over MCP (server.py) and called in-process by the agents.
"""
from __future__ import annotations

import json
import math

import pandas as pd

from . import analysis, governance
from .llm import ToolSpec


def _clean(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


class Toolbox:
    """Audit tools bound to one set of prediction logs."""

    def __init__(self, logs: pd.DataFrame, baseline_months: int = 3, group_col: str = "payer"):
        self.logs = logs
        self.baseline_months = baseline_months
        self.group_col = group_col
        self._metrics = analysis.group_metrics(logs, group_col)

    # ------------------------------------------------------------- tools
    def data_overview(self) -> dict:
        """What the logs contain: months, groups, and patients per group."""
        df = self.logs
        return {"months": sorted(int(m) for m in df["month"].unique()),
                "baseline_months": list(range(self.baseline_months)),
                "patients": len(df),
                "groups": {str(g): int(n) for g, n in df[self.group_col].value_counts().items()},
                "model_flags_top_share": round(float(df["flagged"].mean()), 3)}

    def detect_fairness_drift(self) -> dict:
        """Statistical tests for groups whose fairness got worse after the baseline."""
        alerts = analysis.detect_drift(self.logs, self.baseline_months, group_col=self.group_col)
        return {"alerts": [a.to_dict() for a in alerts],
                "method": "two-proportion and calibration z-tests vs baseline, Bonferroni-corrected, "
                          "minimum effect size, must persist 2 months"}

    def fairness_trend(self, group: str) -> dict:
        """Month-by-month fairness metrics for one group."""
        m = self._metrics[self._metrics["group"] == group]
        if m.empty:
            return {"error": f"unknown group {group!r}"}
        cols = ["month", "n", "tpr", "calibration_gap", "selection_rate", "observed_rate"]
        return {"group": group, "trend": [{k: (round(float(v), 4) if k not in ("month", "n") else int(v))
                                           for k, v in row.items()} for row in m[cols].to_dict("records")]}

    def find_root_cause(self, group: str, start_month: int) -> dict:
        """Which inputs changed for this group, beyond any change in other groups, and did the real
        outcome change?"""
        rows = analysis.root_cause(self.logs, group, int(start_month), self.baseline_months, self.group_col)
        return {"group": group, "since_month": int(start_month), "features": rows[:-1], "outcome": rows[-1],
                "how_to_read": "group_specific_shift is PSI in this group minus PSI in the others (> 0.2 is a "
                               "big shift). group_specific_outcome_change near 0, or a "
                               "large outcome_change_p_value, means the patients' real readmission rate did "
                               "not change."}

    def lookup_governance(self, cause_type: str) -> dict:
        """Recommended action, owner, and NIST AI RMF controls for a cause type:
        data_quality, population_change, or unclear."""
        return governance.lookup(cause_type)

    # ------------------------------------------------------------- plumbing
    SPECS = {
        "data_overview": ToolSpec("data_overview", "What the prediction logs contain.",
                                  {"type": "object", "properties": {}}),
        "detect_fairness_drift": ToolSpec(
            "detect_fairness_drift", "Run statistical drift tests on per-group fairness metrics.",
            {"type": "object", "properties": {}}),
        "fairness_trend": ToolSpec(
            "fairness_trend", "Month-by-month TPR, calibration gap, and selection rate for one group.",
            {"type": "object", "properties": {"group": {"type": "string"}}, "required": ["group"]}),
        "find_root_cause": ToolSpec(
            "find_root_cause", "Rank the inputs that shifted for one group since a month, and show whether "
                               "the real outcome changed.",
            {"type": "object", "properties": {"group": {"type": "string"}, "start_month": {"type": "integer"}},
             "required": ["group", "start_month"]}),
        "lookup_governance": ToolSpec(
            "lookup_governance", "Get the recommended action and NIST AI RMF controls for a cause type.",
            {"type": "object", "properties": {"cause_type": {"type": "string",
                                                             "enum": ["data_quality", "population_change",
                                                                      "unclear"]}},
             "required": ["cause_type"]}),
    }

    def specs(self, names: list[str]) -> list[ToolSpec]:
        return [self.SPECS[n] for n in names]

    def call(self, name: str, args: dict) -> str:
        if name not in self.SPECS:
            return json.dumps({"error": f"unknown tool {name}"})
        try:
            return json.dumps(_clean(getattr(self, name)(**args)))
        except Exception as e:   # the model sent bad arguments: tell it, don't crash
            return json.dumps({"error": f"{type(e).__name__}: {e}"})
