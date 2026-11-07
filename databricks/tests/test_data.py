"""Config, scenarios, data generation, and data quality rules."""
import inspect
import json

import numpy as np
import pandas as pd
import pytest

from driftwatch import analysis
from driftwatch.simulate import Scenario, generate
from driftwatch_databricks import config, generate as gen, quality, scenarios
from driftwatch_databricks.tasks import generate_data_main


def test_thresholds_match_core_package():
    defaults = {k: v.default for k, v in inspect.signature(analysis.detect_drift).parameters.items()}
    assert defaults["baseline_months"] == config.BASELINE_MONTHS
    assert defaults["min_tpr_drop"] == config.MIN_TPR_DROP
    assert defaults["min_calibration_shift"] == config.MIN_CALIBRATION_SHIFT


@pytest.mark.parametrize("bad", ["", "a b", "x;DROP TABLE y", "a`b", "a.b"])
def test_names_reject_unsafe_identifiers(bad):
    with pytest.raises(ValueError):
        config.Names("workspace", bad)


def test_names_paths():
    n = config.Names("workspace", "dev_me_driftwatch")
    assert n.table("silver_prediction_logs") == "`workspace`.`dev_me_driftwatch`.`silver_prediction_logs`"
    assert n.landing_path == "/Volumes/workspace/dev_me_driftwatch/landing/prediction_logs"


def test_every_scenario_builds_and_unknown_fails():
    for name in scenarios.names():
        assert scenarios.build(name, seed=1).name == name
    with pytest.raises(ValueError, match="unknown scenario"):
        scenarios.build("nope", seed=1)


def test_named_scenario_matches_core_simulator():
    """make_batch must not change the simulator's numbers, only add columns."""
    ours = gen.make_batch("medicaid_undercoding", seed=3, batch_id="b", months=4, patients_per_month=300)
    core = generate(Scenario("x", months=4, patients_per_month=300, seed=3,
                             drifts=scenarios.NAMED["medicaid_undercoding"]))
    for col in ["month", "payer", "prior_admissions", "readmitted", "flagged"]:
        assert (ours[col].to_numpy() == core[col].to_numpy()).all()
    assert np.allclose(ours["score"], core["score"])


def test_batch_columns_and_types(small_batch):
    assert list(small_batch.columns) == list(config.LOG_COLUMNS)
    assert small_batch["batch_id"].nunique() == 1
    assert str(small_batch["month"].dtype) == "int32"
    assert small_batch["score"].between(0, 1).all()


def test_batch_id_and_seed():
    assert gen.make_batch_id("medicaid_undercoding", "12345") == "medicaid_undercoding-12345"
    assert gen.make_batch_id("x y", "1") == "x_y-1"
    assert gen.make_batch_id("none", "local").startswith("none-20")
    assert gen.resolve_seed("42") == 42
    assert isinstance(gen.resolve_seed("auto"), int)


def test_generate_task_writes_parquet(tmp_path, capsys):
    generate_data_main(["--catalog", "workspace", "--schema", "driftwatch", "--scenario", "none",
                        "--seed", "1", "--months", "4", "--patients-per-month", "200",
                        "--run-id", "777", "--landing-dir", str(tmp_path)])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["batch_id"] == "none-777"
    back = pd.read_parquet(out["path"])
    assert len(back) == 800 and list(back.columns) == list(config.LOG_COLUMNS)


# ------------------------------------------------------------------ quality rules
def test_every_rule_has_a_pandas_version(small_batch):
    assert set(quality.check(small_batch)) == set(quality.DROP_RULES) | set(quality.WARN_RULES)


def test_clean_batch_passes_every_rule(small_batch):
    silver, failures = quality.to_silver(small_batch)
    assert len(silver) == len(small_batch)
    assert not any(failures.values())


def test_bad_rows_are_dropped_or_counted(small_batch):
    bad = small_batch.head(6).copy()
    bad.loc[0, "score"] = 1.5            # drop
    bad.loc[1, "readmitted"] = 2         # drop
    bad.loc[2, "payer"] = None           # drop
    bad.loc[3, "age"] = 150.0            # warn only
    bad.loc[4, "score"] = np.nan         # drop
    silver, failures = quality.to_silver(bad)
    assert len(silver) == 2              # rows 3 and 5 stay
    assert failures["score_is_probability"] == 2
    assert failures["readmitted_is_binary"] == 1
    assert failures["keys_present"] == 1
    assert failures["age_in_range"] == 1
