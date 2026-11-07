"""The audit step: run DriftWatch on one batch of silver logs and shape the results as rows.

It reuses the core package as it is:
  - driftwatch.tools.Toolbox and driftwatch.agents.rule_audit (statistics, no LLM), always;
  - driftwatch.agents.agent_audit (the four-agent team), only when an LLM key is available.
If the agent audit fails (no key, no outbound internet, API error) the rules result is kept
and the reason is written to the report. The job never fails because of the LLM.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

from driftwatch import governance
from driftwatch.agents import agent_audit, rule_audit
from driftwatch.tools import Toolbox

from . import scenarios
from .config import BASELINE_MONTHS

# secret key name -> (provider name for driftwatch.llm.make_model, env var the SDK reads)
LLM_SECRETS = {
    "anthropic_api_key": ("anthropic", "ANTHROPIC_API_KEY"),
    "openai_api_key": ("openai", "OPENAI_API_KEY"),
}


@dataclass
class AuditOutput:
    mode: str                         # "rules" or "agents"
    findings: list[dict]
    alerts: list[dict]
    report: str
    planted: list[dict] = field(default_factory=list)
    score: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def find_llm(get_secret, scope: str | None) -> tuple[str, str] | None:
    """Look for an LLM key in a Databricks secret scope.

    get_secret(scope, key) returns the value or raises. Returns (provider, key) or None.
    """
    if not scope:
        return None
    for key, (provider, _env) in LLM_SECRETS.items():
        try:
            value = get_secret(scope, key)
        except Exception:
            continue
        if value:
            return provider, value
    return None


def make_llm(provider: str, api_key: str, model: str | None = None):
    from driftwatch.llm import make_model

    env = {p: e for p, e in LLM_SECRETS.values()}[provider]
    os.environ[env] = api_key           # the OpenAI / Anthropic SDKs read the key from here
    return make_model(provider, model)


def _same(found: dict, expected: dict) -> bool:
    return (found.get("group") == expected["group"]
            and abs(int(found.get("start_month") or -99) - expected["start_month"]) <= 1
            and found.get("cause_type") == expected["cause_type"])


def score_against_planted(findings: list[dict], expected: list[dict]) -> dict:
    """The data is simulated, so we know the right answer. Compare, like bench/run_bench.py."""
    found = sum(any(_same(f, e) for f in findings) for e in expected)
    false = sum(not any(f.get("group") == e["group"] for e in expected) for f in findings)
    return {"planted": len(expected), "found": found, "false": false,
            "fully_correct": found == len(expected) and false == 0}


def run_audit(logs: pd.DataFrame, scenario: str | None = None, baseline_months: int = BASELINE_MONTHS,
              llm=None, group_col: str = "payer") -> AuditOutput:
    """Audit one batch of prediction logs. `llm` is a driftwatch.llm model or None."""
    box = Toolbox(logs, baseline_months, group_col)
    alerts = box.detect_fairness_drift()["alerts"]
    result, mode, notes = rule_audit(box), "rules", []
    if llm is not None:
        try:
            result, mode = agent_audit(box, llm), "agents"
        except Exception as e:   # keep the rules result; say why
            notes.append(f"Agent audit failed, kept the rules-only audit: {type(e).__name__}: {e}")
    else:
        notes.append("No LLM key found, so this is the rules-only audit (no agents).")

    planted, score = [], {}
    if scenario:
        try:
            from driftwatch.simulate import expected_findings
            planted = expected_findings(scenarios.build(scenario, seed=0))
            score = score_against_planted(result.findings, planted)
        except ValueError:
            notes.append(f"Scenario {scenario!r} is not a known simulation, so there is no ground truth.")
    return AuditOutput(mode, result.findings, alerts, result.report, planted, score, notes)


def _controls(f: dict) -> list[str]:
    c = f.get("controls") or []
    if isinstance(c, dict):
        return list(c)
    if isinstance(c, str):
        return [x.strip() for x in c.split(",") if x.strip()]
    return [str(x) for x in c]


def findings_rows(out: AuditOutput, batch_id: str, scenario: str, audited_at: datetime) -> list[dict]:
    rows = []
    for f in out.findings:
        controls = _controls(f)
        rows.append({
            "batch_id": batch_id, "scenario": scenario, "audited_at": audited_at, "audit_mode": out.mode,
            "group": f.get("group"), "start_month": int(f["start_month"]) if f.get("start_month") is not None else None,
            "metrics": ",".join(f.get("metrics") or []), "cause_feature": f.get("cause_feature"),
            "cause_type": f.get("cause_type"), "action": f.get("action"), "owner": f.get("owner"),
            "interim": f.get("interim"), "controls": ",".join(controls),
            "nist_controls_json": json.dumps({c: governance.NIST_AI_RMF.get(c, "") for c in controls}),
            "evidence": f.get("evidence"),
            "matches_planted": any(_same(f, e) for e in out.planted) if out.planted or out.score else None,
        })
    return rows


def alerts_rows(out: AuditOutput, batch_id: str, scenario: str, audited_at: datetime) -> list[dict]:
    return [{"batch_id": batch_id, "scenario": scenario, "audited_at": audited_at, "group": a["group"],
             "metric": a["metric"], "first_month": int(a["first_month"]), "baseline": float(a["baseline"]),
             "current": float(a["current"]), "change": float(a["change"]), "p_value": float(a["p_value"])}
            for a in out.alerts]


def render_report(out: AuditOutput, batch_id: str, scenario: str, audited_at: datetime | None = None) -> str:
    audited_at = audited_at or datetime.now(timezone.utc)
    lines = [f"# DriftWatch fairness audit: batch `{batch_id}`", "",
             f"- Scenario: `{scenario}`", f"- Audit mode: **{out.mode}**",
             f"- Audited at: {audited_at:%Y-%m-%d %H:%M} UTC",
             f"- Statistical alerts: {len(out.alerts)}; incidents reported: {len(out.findings)}", ""]
    for n in out.notes:
        lines.append(f"> {n}")
    if out.notes:
        lines.append("")
    lines += [out.report.strip(), ""]
    if out.score:
        s = out.score
        lines += ["## Check against the planted problem (simulated data only)", "",
                  f"Planted incidents: {s['planted']}. Found: {s['found']}. False incidents: {s['false']}. "
                  f"Fully correct: {'yes' if s['fully_correct'] else 'no'}.", ""]
    return "\n".join(lines)


def report_row(out: AuditOutput, batch_id: str, scenario: str, audited_at: datetime, report_md: str) -> dict:
    return {"batch_id": batch_id, "scenario": scenario, "audited_at": audited_at, "audit_mode": out.mode,
            "planted_incidents": out.score.get("planted"), "found_incidents": out.score.get("found"),
            "false_incidents": out.score.get("false"), "notes": " ".join(out.notes), "report_md": report_md}
