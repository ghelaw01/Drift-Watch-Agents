"""The audit team: Monitor -> Investigator -> Governance -> Reporter.

Each agent has its own tools and its own job. Python passes structured JSON between them,
so every handoff can be checked. Agents decide and explain; tools compute every number.

`rule_audit` does the same job with fixed rules and no LLM. It is the baseline the agents
must match, and the fallback when no model is available.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

from . import governance
from .llm import Reply
from .tools import Toolbox

CAUSE_TYPES = ("data_quality", "population_change", "unclear")


@dataclass
class Step:
    agent: str
    tool: str
    args: dict


@dataclass
class AuditResult:
    findings: list[dict]
    report: str
    steps: list[Step] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def to_dict(self):
        return {"findings": self.findings, "report": self.report,
                "steps": [s.__dict__ for s in self.steps], "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "seconds": round(self.seconds, 2)}


def parse_json(text: str) -> dict:
    """Take the last JSON object in a model reply (inside ```json fences or not)."""
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.S)
    candidates = fenced or re.findall(r"\{.*\}", text or "", re.S)
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return {}


class Agent:
    def __init__(self, name: str, instructions: str, model, toolbox: Toolbox, tools: list[str],
                 max_steps: int = 8):
        self.name, self.instructions, self.model = name, instructions, model
        self.toolbox, self.tools, self.max_steps = toolbox, tools, max_steps

    def run(self, task: str, result: AuditResult) -> str:
        messages: list[dict] = [{"role": "user", "content": task}]
        specs = self.toolbox.specs(self.tools)
        for _ in range(self.max_steps):
            reply: Reply = self.model.chat(self.instructions, messages, specs)
            result.input_tokens += reply.input_tokens
            result.output_tokens += reply.output_tokens
            if not reply.tool_calls:
                return reply.text or ""
            messages.append({"role": "assistant", "content": reply.text, "tool_calls": reply.tool_calls})
            for call in reply.tool_calls:
                result.steps.append(Step(self.name, call.name, call.args))
                messages.append({"role": "tool", "id": call.id, "name": call.name,
                                 "content": self.toolbox.call(call.name, call.args)})
        messages.append({"role": "user", "content": "Step limit reached. Give your final JSON answer now."})
        return self.model.chat(self.instructions, messages, []).text or ""


MONITOR = """You are the Monitor in a fairness audit of a deployed hospital readmission model.
The model flags high-risk patients for follow-up care. Groups are insurance payers.
Use your tools to find groups whose fairness got significantly worse after the baseline months.
Rules:
- Report an incident only if detect_fairness_drift raised an alert for that group.
- One incident per group. Use the earliest alert month as start_month.
- Use fairness_trend to confirm the change is real and lasting before you report it.
- If there are no alerts, report no incidents. Do not guess.
End with JSON only, in this shape:
```json
{"incidents": [{"group": "...", "start_month": 0, "metrics": ["tpr"], "summary": "one sentence with numbers"}]}
```"""

INVESTIGATOR = """You are the Investigator in a fairness audit of a hospital readmission model.
You get one incident. Find out why it happened, using find_root_cause.
Decide the cause type:
- population_change: an input shifted AND the group's real readmission rate rose significantly
  (group_specific_outcome_change of 0.02 or more AND outcome_change_p_value under 0.01). The patients really changed.
- data_quality: an input feature shifted for this group, but the group's real readmission rate did not
  change significantly. The data describing the patients is wrong, not the patients.
- unclear: the evidence does not support either.
cause_feature is the feature with the largest group_specific_shift.
End with JSON only:
```json
{"cause_feature": "...", "cause_type": "data_quality", "evidence": "one or two sentences with numbers"}
```"""

GOVERNANCE = """You are the Governance officer in a fairness audit of a hospital readmission model.
You get one investigated incident. Call lookup_governance with its cause type, then decide.
Use the playbook action unless the evidence clearly says otherwise.
End with JSON only:
```json
{"action": "...", "controls": ["MEASURE 2.11"], "owner": "...", "interim": "...", "rationale": "one sentence"}
```"""

REPORTER = """You write the fairness incident report for the hospital's AI governance committee.
Use ONLY the facts and numbers in the findings you are given. Do not add new numbers.
Write clear Markdown: a one-paragraph summary, then one section per incident
(what happened, why, what to do now, owner, NIST AI RMF controls), then "Checked and not flagged".
If there are no incidents, say so plainly and say what was checked."""


def agent_audit(toolbox: Toolbox, model) -> AuditResult:
    """Run the four-agent audit team."""
    start = time.time()
    result = AuditResult([], "")
    monitor = Agent("monitor", MONITOR, model, toolbox, ["data_overview", "detect_fairness_drift", "fairness_trend"])
    incidents = parse_json(monitor.run("Audit these prediction logs for fairness drift.", result)).get("incidents", [])

    # Coverage check, in code: every group with a statistical alert must be investigated.
    # In testing, a model sometimes reported only the first of two incidents.
    reported = {i.get("group") for i in incidents}
    for a in toolbox.detect_fairness_drift()["alerts"]:
        if a["group"] not in reported:
            reported.add(a["group"])
            incidents.append({"group": a["group"], "start_month": a["first_month"], "metrics": [a["metric"]],
                              "summary": f"{a['metric']} changed by {a['change']} (added by coverage check)"})

    for inc in incidents:
        finding = {"group": inc.get("group"), "start_month": inc.get("start_month"),
                   "metrics": inc.get("metrics", []), "summary": inc.get("summary", "")}
        inv = parse_json(Agent("investigator", INVESTIGATOR, model, toolbox, ["find_root_cause", "fairness_trend"])
                         .run(f"Incident: {json.dumps(finding)}", result))
        finding.update({"cause_feature": inv.get("cause_feature"),
                        "cause_type": inv.get("cause_type") if inv.get("cause_type") in CAUSE_TYPES else "unclear",
                        "evidence": inv.get("evidence", "")})
        gov = parse_json(Agent("governance", GOVERNANCE, model, toolbox, ["lookup_governance"])
                         .run(f"Investigated incident: {json.dumps(finding)}", result))
        finding.update({k: gov.get(k) for k in ("action", "controls", "owner", "interim", "rationale")})
        result.findings.append(finding)

    checked = toolbox.data_overview()
    result.report = Agent("reporter", REPORTER, model, toolbox, []).run(
        f"Findings: {json.dumps(result.findings, indent=1)}\nData checked: {json.dumps(checked)}", result)
    result.seconds = time.time() - start
    return result


def rule_audit(toolbox: Toolbox, outcome_threshold: float = 0.02) -> AuditResult:
    """The same audit with fixed rules and no LLM."""
    start = time.time()
    alerts = toolbox.detect_fairness_drift()["alerts"]
    first: dict[str, dict] = {}
    for a in alerts:
        f = first.setdefault(a["group"], {"group": a["group"], "start_month": a["first_month"], "metrics": []})
        f["start_month"] = min(f["start_month"], a["first_month"])
        f["metrics"].append(a["metric"])
    findings = []
    for f in first.values():
        rc = toolbox.find_root_cause(f["group"], f["start_month"])
        top = rc["features"][0]
        outcome = rc["outcome"]["group_specific_outcome_change"]
        real = outcome >= outcome_threshold and rc["outcome"]["outcome_change_p_value"] < 0.01
        cause = "population_change" if real else "data_quality"
        play = governance.lookup(cause)
        findings.append({**f, "cause_feature": top["feature"], "cause_type": cause,
                         "evidence": f"{top['feature']} group-specific PSI {top['group_specific_shift']}; "
                                     f"outcome change {outcome}",
                         "action": play["action"], "controls": list(play["controls"]),
                         "owner": play["owner"], "interim": play["interim"]})
    lines = ["# Fairness audit (rules only)", ""]
    lines += [f"- **{f['group']}** from month {f['start_month']}: {f['cause_type']} via {f['cause_feature']}"
              f" -> {f['action']} ({', '.join(f['controls'])})" for f in findings] or ["No fairness drift found."]
    return AuditResult(findings, "\n".join(lines), seconds=time.time() - start)
