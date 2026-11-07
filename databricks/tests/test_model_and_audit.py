"""Gold metrics, training, and the audit step (pandas only, no Spark)."""
import json
from datetime import datetime

import numpy as np
import pytest

from driftwatch import analysis
from driftwatch.llm import Reply
from driftwatch_databricks import audit, config, metrics, training

NOW = datetime(2026, 1, 1, 12, 0)


def _ddl_columns(ddl: str) -> list[str]:
    return [part.strip().split(" ")[0].strip("`") for part in ddl.split(",")]


# ------------------------------------------------------------------ gold
def test_gold_metrics_equal_core_group_metrics(undercoded_batch):
    gold = metrics.gold_metrics(undercoded_batch)
    core = analysis.group_metrics(undercoded_batch)
    assert len(gold) == len(core) == 12 * 4
    merged = gold.merge(core, on=["month", "group"], suffixes=("", "_core"))
    for col in ["tpr", "selection_rate", "calibration_gap"]:
        assert np.allclose(merged[col], merged[f"{col}_core"])


def test_watchlist_flags_the_planted_group(undercoded_batch):
    w = metrics.watchlist(undercoded_batch)
    assert not w.loc[w["in_baseline"], ["tpr_drop_flag", "calibration_flag"]].any().any()
    late = w[(w["group"] == "Medicaid") & (w["month"] >= 7)]
    assert late["tpr_drop_flag"].all()


# ------------------------------------------------------------------ training
def test_baseline_training_reproduces_the_deployed_model(undercoded_batch):
    r = training.train(undercoded_batch, window="baseline")
    assert r.train_months == [0, 1, 2]
    from driftwatch.simulate import FEATURES
    score = r.model.predict_proba(undercoded_batch[FEATURES])[:, 1]
    assert np.abs(score.round(4) - undercoded_batch["score"].to_numpy()).max() <= 1e-4
    flags = (score >= r.threshold).astype(int)
    assert (flags != undercoded_batch["flagged"]).mean() < 0.001     # rounding at the threshold only
    assert 0.6 < r.metrics["train_auc"] < 0.95
    assert "recent_medicaid_tpr" in r.metrics and "train_max_tpr_gap" in r.metrics
    assert set(r.fairness["window"]) == {"train", "recent"}


def test_recent_window_and_promotion_gate(undercoded_batch):
    r = training.train(undercoded_batch, window="recent")
    assert r.train_months == [9, 10, 11]
    assert training.should_promote(r, 0.5)[0] is True
    assert training.should_promote(r, 0.99)[0] is False
    with pytest.raises(ValueError):
        training.train(undercoded_batch, window="all")


def test_log_to_mlflow_locally(undercoded_batch, tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_DISABLE_AGENT_HINT", "1")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")     # local test only; Databricks uses its own store
    mlflow.set_tracking_uri(f"file://{tmp_path}/mlruns")
    r = training.train(undercoded_batch)
    out = training.log_to_mlflow(r, experiment="driftwatch-test", registered_model=None)
    run = mlflow.get_run(out["run_id"])
    assert run.data.params["window"] == "baseline"
    assert run.data.metrics["train_auc"] == pytest.approx(r.metrics["train_auc"])
    assert out["promoted"] is False                 # nothing registered
    model = mlflow.pyfunc.load_model(out["model_uri"])
    pred = np.asarray(model.predict(r.input_example))
    assert pred.shape == (5, 2)                     # predict_proba, as served


# ------------------------------------------------------------------ audit
def test_rules_audit_finds_the_planted_problem(undercoded_batch):
    out = audit.run_audit(undercoded_batch, scenario="medicaid_undercoding")
    assert out.mode == "rules"
    assert [f["group"] for f in out.findings] == ["Medicaid"]
    assert out.findings[0]["cause_type"] == "data_quality"
    assert out.score == {"planted": 1, "found": 1, "false": 0, "fully_correct": True}
    assert any("No LLM key" in n for n in out.notes)


def test_clean_logs_give_no_incident(clean_batch):
    out = audit.run_audit(clean_batch, scenario="none")
    assert out.findings == [] and out.score["fully_correct"]


def test_rows_match_table_schemas(undercoded_batch):
    out = audit.run_audit(undercoded_batch, scenario="medicaid_undercoding")
    f = audit.findings_rows(out, "b1", "medicaid_undercoding", NOW)
    assert list(f[0]) == _ddl_columns(config.FINDINGS_DDL)
    assert f[0]["matches_planted"] is True
    controls = json.loads(f[0]["nist_controls_json"])
    assert "MEASURE 2.11" in controls and controls["MEASURE 2.11"]
    a = audit.alerts_rows(out, "b1", "medicaid_undercoding", NOW)
    assert a and list(a[0]) == _ddl_columns(config.ALERTS_DDL)
    report = audit.render_report(out, "b1", "medicaid_undercoding", NOW)
    assert "Medicaid" in report and "Fully correct: yes" in report
    r = audit.report_row(out, "b1", "medicaid_undercoding", NOW, report)
    assert list(r) == _ddl_columns(config.REPORTS_DDL)


def test_unknown_scenario_has_no_ground_truth(clean_batch):
    out = audit.run_audit(clean_batch, scenario="my-real-logs")
    assert out.score == {} and out.planted == []
    assert audit.findings_rows(out, "b", "s", NOW) == []


class _Broken:
    def chat(self, *a, **k):
        raise ConnectionError("no outbound internet")


class _Silent:
    """A model that answers every agent with an empty JSON object."""
    def chat(self, system, messages, tools):
        return Reply('{"incidents": []}', [], 10, 5)


def test_llm_failure_falls_back_to_rules(undercoded_batch):
    out = audit.run_audit(undercoded_batch, scenario="medicaid_undercoding", llm=_Broken())
    assert out.mode == "rules" and len(out.findings) == 1
    assert any("ConnectionError" in n for n in out.notes)


def test_agent_mode_runs_and_coverage_check_keeps_the_incident(undercoded_batch):
    out = audit.run_audit(undercoded_batch, scenario="medicaid_undercoding", llm=_Silent())
    assert out.mode == "agents"
    assert [f["group"] for f in out.findings] == ["Medicaid"]      # added by the coverage check
    rows = audit.findings_rows(out, "b", "s", NOW)
    assert rows[0]["controls"] == ""                               # the silent model chose none


def test_find_llm_uses_the_first_key_found():
    store = {("dw", "openai_api_key"): "sk-test"}

    def getter(scope, key):
        return store[(scope, key)]
    assert audit.find_llm(getter, "dw") == ("openai", "sk-test")
    assert audit.find_llm(getter, "other") is None
    assert audit.find_llm(getter, "") is None
