"""The agent pipeline with a scripted fake model. No API key needed."""
import json

from driftwatch import Toolbox, agent_audit, rule_audit, score
from driftwatch.agents import parse_json
from driftwatch.llm import Reply, ToolCall

EXPECTED = [{"group": "Medicaid", "start_month": 6, "cause_feature": "prior_admissions",
             "cause_type": "data_quality", "action": "fix_data_pipeline"}]


class ScriptedModel:
    """Calls the right tools, then answers from what the tools returned."""

    def chat(self, system, messages, tools):
        last = messages[-1]
        names = [t.name for t in tools]
        if last["role"] == "user" and names:
            first_tool = {"detect_fairness_drift": "detect_fairness_drift", "find_root_cause": "find_root_cause",
                          "lookup_governance": "lookup_governance"}
            tool = next(n for n in names if n in first_tool)
            args = {}
            if tool == "find_root_cause":
                inc = json.loads(last["content"].split("Incident: ", 1)[1])
                args = {"group": inc["group"], "start_month": inc["start_month"]}
            if tool == "lookup_governance":
                args = {"cause_type": json.loads(last["content"].split(": ", 1)[1])["cause_type"]}
            return Reply(None, [ToolCall("c1", tool, args)])
        if not names:
            return Reply("# Report\nMedicaid: fix the data pipeline.")
        out = json.loads(last["content"])
        if "alerts" in out:
            groups = {}
            for a in out["alerts"]:
                groups.setdefault(a["group"], a["first_month"])
            ans = {"incidents": [{"group": g, "start_month": m, "metrics": ["tpr"], "summary": "TPR fell"}
                                 for g, m in groups.items()]}
        elif "features" in out:
            change = out["outcome"]["group_specific_outcome_change"]
            ans = {"cause_feature": out["features"][0]["feature"],
                   "cause_type": "population_change" if change >= 0.02 else "data_quality", "evidence": "x"}
        else:
            ans = {"action": out["action"], "controls": list(out["controls"]), "owner": out["owner"],
                   "interim": out["interim"], "rationale": "playbook"}
        return Reply(f"Done.\n```json\n{json.dumps(ans)}\n```")


def test_agent_team_finds_cause_and_action(undercoded):
    result = agent_audit(Toolbox(undercoded), ScriptedModel())
    s = score(result.findings, EXPECTED)
    assert s["fully_correct"], s
    assert [st.agent for st in result.steps] == ["monitor", "investigator", "governance"]
    assert "MEASURE 2.11" in result.findings[0]["controls"]
    assert result.report.startswith("# Report")


def test_rules_baseline_matches_ground_truth(undercoded, older_population, clean):
    assert score(rule_audit(Toolbox(undercoded)).findings, EXPECTED)["fully_correct"]
    pop = rule_audit(Toolbox(older_population)).findings
    assert pop[0]["cause_type"] == "population_change" and pop[0]["action"] == "recalibrate_or_retrain"
    assert rule_audit(Toolbox(clean)).findings == []


def test_score_counts_false_positives_and_misses():
    s = score([{"group": "Private", "start_month": 4}], EXPECTED)
    assert s["false_pos"] == 1 and s["missed"] == 1 and not s["fully_correct"]


def test_parse_json_takes_last_block_and_survives_garbage():
    assert parse_json('x ```json\n{"a": 1}\n``` then ```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json("no json here") == {}


def test_bad_tool_arguments_return_an_error_not_a_crash(clean):
    out = json.loads(Toolbox(clean).call("find_root_cause", {"group": "Medicaid"}))
    assert "error" in out
