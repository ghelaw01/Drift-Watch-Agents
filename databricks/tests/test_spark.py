"""Spark parity tests. Skipped when pyspark (or Java) is not available.

They check that the PySpark code used by the pipeline gives the same answers as the
pandas code that the other tests cover.
"""
import os

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark")

from driftwatch_databricks import config, generate, metrics, quality, spark_gold  # noqa: E402
from driftwatch_databricks.tasks import rows_to_frame  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    from pyspark.sql import SparkSession

    try:
        s = (SparkSession.builder.master("local[1]").appName("driftwatch-tests")
             .config("spark.sql.shuffle.partitions", "2").config("spark.ui.enabled", "false")
             .config("spark.sql.session.timeZone", "UTC").config("spark.sql.ansi.enabled", "true")
             .getOrCreate())
    except Exception as e:  # no Java, etc.
        pytest.skip(f"Spark could not start: {e}")
    yield s
    s.stop()


@pytest.fixture(scope="module")
def bronze(spark, tmp_path_factory, small_batch):
    """The batch written as Parquet (like the generate task) and read back by Spark."""
    d = tmp_path_factory.mktemp("landing")
    generate.write_batch(small_batch, str(d))
    return spark.read.parquet(str(d))


def test_parquet_types_are_readable(bronze):
    types = dict(bronze.dtypes)
    assert types["generated_at"] == "timestamp"
    assert types["month"] == "int" and types["score"] == "double"


def test_quality_sql_matches_pandas(spark, small_batch):
    bad = small_batch.head(8).copy()
    bad.loc[0, "score"] = 1.5
    bad.loc[1, "readmitted"] = 2
    bad.loc[2, "payer"] = None
    bad.loc[3, "age"] = 150.0
    bad.loc[4, "score"] = np.nan
    bad.loc[5, "num_medications"] = -1
    sdf = spark.createDataFrame(bad)
    pandas_ok = quality.check(bad)
    for name, expr in {**quality.DROP_RULES, **quality.WARN_RULES}.items():
        from pyspark.sql import functions as F
        got = [bool(r[0]) for r in sdf.select(F.coalesce(F.expr(expr), F.lit(False))).collect()]
        assert got == pandas_ok[name].tolist(), name


def test_silver_cast_and_gold_metrics_match_pandas(bronze, small_batch):
    from pyspark.sql import functions as F

    silver = bronze.select(*[F.col(c).cast(t).alias(c) for c, t in config.LOG_COLUMNS.items()])
    got = spark_gold.gold_metrics(silver).toPandas().sort_values(["group", "month"]).reset_index(drop=True)
    want = metrics.gold_metrics(small_batch).sort_values(["group", "month"]).reset_index(drop=True)
    assert len(got) == len(want)
    for col in ["n", "positives", "tpr", "selection_rate", "observed_rate", "predicted_rate", "calibration_gap"]:
        assert np.allclose(got[col].astype(float), want[col].astype(float), equal_nan=True), col


def test_watchlist_matches_pandas(bronze, small_batch):
    got = spark_gold.watchlist(bronze, baseline_months=2).toPandas()
    want = metrics.watchlist(small_batch, baseline_months=2)
    key = ["group", "month"]
    got, want = got.sort_values(key).reset_index(drop=True), want.sort_values(key).reset_index(drop=True)
    for col in ["baseline_tpr", "tpr_change", "calibration_change"]:
        assert np.allclose(got[col].astype(float), want[col].astype(float), equal_nan=True), col
    for col in ["tpr_drop_flag", "calibration_flag", "in_baseline"]:
        assert got[col].tolist() == want[col].tolist(), col


def test_rows_to_frame_follows_the_ddl(spark):
    empty = rows_to_frame(spark, [], config.ALERTS_DDL)
    assert empty.count() == 0 and empty.columns[3] == "group"
    from datetime import datetime
    row = {"batch_id": "b", "scenario": "s", "audited_at": datetime(2026, 1, 1), "group": "Medicaid",
           "metric": "tpr", "first_month": 6, "baseline": 0.5, "current": 0.3, "change": -0.2, "p_value": 1e-9}
    df = rows_to_frame(spark, [dict(reversed(list(row.items())))], config.ALERTS_DDL)
    assert df.collect()[0]["group"] == "Medicaid" and df.columns == list(row)


def test_findings_rows_fit_the_findings_ddl(spark, small_batch):
    from datetime import datetime

    from driftwatch_databricks import audit

    out = audit.AuditOutput("rules", [{"group": "Medicaid", "start_month": 6, "metrics": ["tpr"],
                                       "cause_feature": "prior_admissions", "cause_type": "data_quality",
                                       "action": "fix_data_pipeline", "controls": ["MEASURE 2.11"]}],
                            [], "report")
    rows = audit.findings_rows(out, "b", "s", datetime(2026, 1, 1))
    df = rows_to_frame(spark, rows, config.FINDINGS_DDL)
    assert df.collect()[0]["controls"] == "MEASURE 2.11"


def test_pipeline_file_runs_with_a_stand_in_dlt(spark, bronze, small_batch, monkeypatch):
    """Load src/pipeline/driftwatch_pipeline.py with a fake `dlt` module and run its gold tables.

    Auto Loader and streaming need Databricks, so bronze and silver are only checked for their
    declarations. The gold functions really run, on temp views named like the pipeline tables.
    """
    import builtins
    import importlib.util
    import sys
    import types
    from pathlib import Path

    tables, expectations = {}, {}

    def table(**kw):
        def deco(fn):
            tables[kw["name"]] = fn
            return fn
        return deco

    def expect_factory(kind):
        def outer(rules_or_name, expr=None):
            def deco(fn):
                rules = rules_or_name if isinstance(rules_or_name, dict) else {rules_or_name: expr}
                expectations.setdefault(fn.__name__, []).append((kind, rules))
                return fn
            return deco
        return outer

    fake = types.SimpleNamespace(table=table, expect=expect_factory("warn"),
                                 expect_all=expect_factory("warn"), expect_all_or_drop=expect_factory("drop"))
    monkeypatch.setitem(sys.modules, "dlt", fake)
    spark.conf.set("driftwatch.landing_path", "/tmp/unused")
    monkeypatch.setattr(builtins, "spark", spark, raising=False)   # the pipeline runtime provides `spark`

    path = Path(__file__).resolve().parents[1] / "src" / "pipeline" / "driftwatch_pipeline.py"
    spec = importlib.util.spec_from_file_location("driftwatch_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert set(tables) == {config.BRONZE, config.SILVER, config.GOLD_METRICS, config.GOLD_WATCHLIST}
    kinds = dict(expectations["silver_prediction_logs"][::-1])
    assert kinds["drop"] == quality.DROP_RULES and kinds["warn"] == quality.WARN_RULES

    bronze.createOrReplaceTempView(config.SILVER)
    gold = tables[config.GOLD_METRICS]().toPandas()
    assert len(gold) == len(metrics.gold_metrics(small_batch))
    watch = tables[config.GOLD_WATCHLIST]().toPandas()
    assert watch["baseline_tpr"].notna().all()


if os.name == "nt":  # pragma: no cover
    pytestmark = pytest.mark.skip("local Spark tests are not run on Windows")
