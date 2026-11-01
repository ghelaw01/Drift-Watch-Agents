"""Score an audit against ground truth."""
from __future__ import annotations


def score(findings: list[dict], expected: list[dict]) -> dict:
    """Match findings to expected incidents by group, then check each field."""
    by_group = {f.get("group"): f for f in findings}
    want = {e["group"]: e for e in expected}
    tp = [g for g in want if g in by_group]
    out = {"expected": len(want), "reported": len(by_group), "true_pos": len(tp),
           "false_pos": len([g for g in by_group if g not in want]), "missed": len(want) - len(tp),
           "month_exact": 0, "month_within_1": 0, "cause_feature": 0, "cause_type": 0, "action": 0,
           "delay_months": []}
    for g in tp:
        f, e = by_group[g], want[g]
        try:
            delay = int(f.get("start_month")) - e["start_month"]
        except (TypeError, ValueError):
            delay = None
        if delay is not None:
            out["delay_months"].append(delay)
            out["month_exact"] += delay == 0
            out["month_within_1"] += abs(delay) <= 1
        out["cause_feature"] += f.get("cause_feature") == e["cause_feature"]
        out["cause_type"] += f.get("cause_type") == e["cause_type"]
        out["action"] += f.get("action") == e["action"]
    out["fully_correct"] = (out["false_pos"] == 0 and out["missed"] == 0 and
                            all(out[k] == len(tp) for k in ("month_within_1", "cause_feature", "cause_type", "action")))
    return out
