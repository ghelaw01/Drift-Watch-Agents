"""Summarize benchmark results and draw the README chart."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "bench" / "results"


def summarize(rows):
    s = [r["score"] for r in rows if "score" in r]
    exp, tp = sum(x["expected"] for x in s), sum(x["true_pos"] for x in s)
    fp = sum(x["false_pos"] for x in s)
    clean = [r for r in rows if "score" in r and r["score"]["expected"] == 0]
    delays = [d for x in s for d in x["delay_months"]]
    return {
        "audits": len(rows), "errors": sum("error" in r for r in rows),
        "fully_correct": sum(x["fully_correct"] for x in s),
        "incidents_found": f"{tp}/{exp}", "false_incidents": fp,
        "clean_audits_with_alarm": f"{sum(r['score']['reported'] > 0 for r in clean)}/{len(clean)}",
        "start_month_within_1": f"{sum(x['month_within_1'] for x in s)}/{tp}",
        "cause_feature": f"{sum(x['cause_feature'] for x in s)}/{tp}",
        "cause_type": f"{sum(x['cause_type'] for x in s)}/{tp}",
        "action": f"{sum(x['action'] for x in s)}/{tp}",
        "mean_delay_months": round(sum(delays) / len(delays), 2) if delays else None,
        "tool_calls": round(sum(r.get("tool_calls", 0) for r in rows) / max(1, len(rows)), 1),
        "seconds": round(sum(r.get("seconds", 0) for r in rows) / max(1, len(rows)), 1),
    }


def table(summaries):
    cols = ["fully_correct", "incidents_found", "false_incidents", "clean_audits_with_alarm",
            "start_month_within_1", "cause_feature", "cause_type", "action", "tool_calls", "seconds"]
    heads = ["Audits fully correct", "Incidents found", "False incidents", "Clean logs with an alarm",
             "Start month (±1)", "Cause feature", "Cause type", "Action", "Tool calls / audit", "Seconds / audit"]
    lines = ["| System | " + " | ".join(heads) + " |", "| --- " * (len(heads) + 1) + "|"]
    for name, s in summaries.items():
        lines.append(f"| {name} | {s['fully_correct']}/{s['audits']} | " + " | ".join(str(s[c]) for c in cols[1:]) + " |")
    return "\n".join(lines)


def chart():
    from driftwatch.analysis import detect_drift, group_metrics
    from driftwatch.simulate import benchmark_suite, generate
    sc = next(s for s in benchmark_suite() if s.name == "drift-9")
    df = generate(sc)
    m = group_metrics(df)
    alerts = detect_drift(df)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharex=True)
    colors = {"Medicaid": "#c2410c", "Medicare": "#1d4ed8", "Private": "#6b7280", "Uninsured": "#9ca3af"}
    for ax, metric, title in ((axes[0], "tpr", "Share of readmitted patients the model flagged (TPR)"),
                              (axes[1], "calibration_gap", "Real risk minus predicted risk")):
        for g, part in m.groupby("group"):
            ax.plot(part["month"], part[metric], marker="o", ms=3, lw=2 if g in ("Medicaid", "Medicare") else 1,
                    color=colors[g], label=g)
        for a in alerts:
            if a.metric == metric:
                ax.axvline(a.first_month, color=colors[a.group], ls="--", lw=1)
                ax.annotate(f"alert: {a.group}", (a.first_month, ax.get_ylim()[1]), fontsize=8,
                            color=colors[a.group], ha="left", va="top", xytext=(3, -3), textcoords="offset points")
        ax.axvspan(-0.5, 2.5, color="#e5e7eb", alpha=0.6)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("month (grey = training period)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Two planted problems: Medicaid records lose prior admissions (month 6); "
                 "an older Medicare population arrives (month 8)", fontsize=10)
    fig.tight_layout()
    (ROOT / "docs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "docs" / "drift-example.png", dpi=150)


if __name__ == "__main__":
    summaries = {}
    for path in sorted(RESULTS.glob("*.jsonl"), key=lambda p: (p.stem != "rules", p.stem)):
        rows = [json.loads(l) for l in path.read_text().splitlines()]
        summaries["rules only (no LLM)" if path.stem == "rules" else f"agent team: {path.stem}"] = summarize(rows)
    text = table(summaries)
    (RESULTS / "summary.md").write_text(text + "\n")
    print(text)
    chart()
