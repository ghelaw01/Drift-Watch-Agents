"""Entry points for the job tasks (python_wheel_task). See resources/job.yml.

Each task is a thin wrapper: parse arguments, read or write Unity Catalog, and call the
pure functions in generate.py, training.py, and audit.py. Spark, MLflow, and the Databricks
SDK are imported inside the functions, so the module imports fine without them.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime, timezone

from .config import ALERTS, ALERTS_DDL, FINDINGS, FINDINGS_DDL, REPORTS, REPORTS_DDL, SILVER, Names

warnings.filterwarnings("ignore")
log = logging.getLogger("driftwatch")


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)


def _spark():
    from pyspark.sql import SparkSession

    return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()


def _common(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    p.add_argument("--catalog", required=True)
    p.add_argument("--schema", required=True)
    p.add_argument("--volume", default="landing")
    return p


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def rows_to_frame(spark, rows: list[dict], ddl: str):
    """A Spark DataFrame with exactly the DDL columns, in DDL order. Works for an empty list."""
    schema = spark.createDataFrame([], ddl).schema
    return spark.createDataFrame([tuple(r.get(f.name) for f in schema.fields) for r in rows], schema)


def _write(spark, rows: list[dict], table: str, ddl: str, mode: str = "append"):
    """Append rows to a Delta table. An empty list still creates the table with its schema."""
    rows_to_frame(spark, rows, ddl).write.format("delta").mode(mode).saveAsTable(table)


def _load_batch(spark, names: Names, batch_id: str):
    """Silver rows for one batch as pandas. batch_id 'latest' picks the newest batch."""
    from pyspark.sql import functions as F

    silver = spark.table(names.table(SILVER))
    if batch_id in ("", "latest", None):
        row = silver.orderBy(F.col("generated_at").desc()).select("batch_id").limit(1).collect()
        if not row:
            raise RuntimeError(f"{names.table(SILVER)} is empty. Run the generate and pipeline tasks first.")
        batch_id = row[0]["batch_id"]
    pdf = silver.where(F.col("batch_id") == batch_id).toPandas()
    if pdf.empty:
        raise RuntimeError(f"No silver rows for batch {batch_id!r}. Did the pipeline run after generate_data?")
    return batch_id, pdf


# ------------------------------------------------------------------ 1. generate data
def generate_data_main(argv=None):
    """Simulate prediction logs with a planted drift scenario and land them in the volume."""
    from . import generate

    _setup_logging()
    p = _common(argparse.ArgumentParser("dw-generate-data"))
    p.add_argument("--scenario", default="medicaid_undercoding")
    p.add_argument("--seed", default="auto")
    p.add_argument("--months", type=int, default=12)
    p.add_argument("--patients-per-month", type=int, default=4000)
    p.add_argument("--run-id", default="local")
    p.add_argument("--landing-dir", default="", help="override the volume path (local runs and tests)")
    a = p.parse_args(argv)
    names = Names(a.catalog, a.schema, a.volume)

    seed = generate.resolve_seed(a.seed)
    batch_id = generate.make_batch_id(a.scenario, a.run_id)
    df = generate.make_batch(a.scenario, seed, batch_id, a.months, a.patients_per_month)
    path = generate.write_batch(df, a.landing_dir or names.landing_path)
    log.info("wrote %d rows for scenario %s (seed %d) to %s", len(df), a.scenario, seed, path)
    print(json.dumps({"batch_id": batch_id, "rows": len(df), "path": path, "seed": seed}))


# ------------------------------------------------------------------ 2. train and register
def train_model_main(argv=None):
    """Train the readmission model on silver data, log it to MLflow, register it in UC."""
    from .training import log_to_mlflow, train

    _setup_logging()
    p = _common(argparse.ArgumentParser("dw-train-model"))
    p.add_argument("--model-name", required=True, help="model name inside the schema")
    p.add_argument("--experiment", default="")
    p.add_argument("--batch-id", default="latest")
    p.add_argument("--window", default="baseline", choices=["baseline", "recent"])
    p.add_argument("--min-auc", type=float, default=0.6)
    a = p.parse_args(argv)
    names = Names(a.catalog, a.schema, a.volume)

    import mlflow

    if not os.getenv("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri("databricks")       # the workspace's MLflow, not a local folder
    batch_id, logs = _load_batch(_spark(), names, a.batch_id)
    result = train(logs, window=a.window)
    full_name = f"{a.catalog}.{a.schema}.{a.model_name}"
    out = log_to_mlflow(result, experiment=a.experiment or None, registered_model=full_name,
                        min_auc=a.min_auc, tags={"batch_id": batch_id, "scenario": str(logs["scenario"].iloc[0])})
    log.info("model %s version %s; promoted to champion: %s (%s)", full_name, out.get("version"),
             out["promoted"], out["reason"])
    print(json.dumps(out))


# ------------------------------------------------------------------ 3. audit
def _secret_getter():
    """dbutils.secrets.get on Databricks; None when it is not available."""
    try:
        from databricks.sdk.runtime import dbutils   # available on serverless jobs

        return dbutils.secrets.get
    except Exception:
        pass
    try:
        import base64

        from databricks.sdk import WorkspaceClient

        w = WorkspaceClient()
        return lambda scope, key: base64.b64decode(w.secrets.get_secret(scope, key).value).decode()
    except Exception:
        return None


def _grant_app_access(spark, names: Names, app_name: str):
    """Let the Databricks App's service principal read this schema (best effort)."""
    if not app_name:
        return "no app name given"
    try:
        from databricks.sdk import WorkspaceClient

        sp = WorkspaceClient().apps.get(app_name).service_principal_client_id
    except Exception as e:
        return f"app {app_name!r} not found yet ({type(e).__name__}); deploy the app, then rerun"
    try:
        for stmt in (f"GRANT USE CATALOG ON CATALOG `{names.catalog}` TO `{sp}`",
                     f"GRANT USE SCHEMA, SELECT ON SCHEMA `{names.catalog}`.`{names.schema}` TO `{sp}`"):
            spark.sql(stmt)
        return f"granted read access on {names.catalog}.{names.schema} to app {app_name}"
    except Exception as e:
        return f"could not grant access to the app ({type(e).__name__}: {e})"


def run_audit_main(argv=None):
    """Run DriftWatch on the newest batch and write findings, alerts, and the report."""
    from .audit import alerts_rows, find_llm, findings_rows, make_llm, render_report, report_row, run_audit

    _setup_logging()
    p = _common(argparse.ArgumentParser("dw-run-audit"))
    p.add_argument("--batch-id", default="latest")
    p.add_argument("--agent-audit", default="auto", choices=["auto", "off"])
    p.add_argument("--secret-scope", default="")
    p.add_argument("--llm-model", default="")
    p.add_argument("--app-name", default="")
    p.add_argument("--fail-on-incident", default="false")
    a = p.parse_args(argv)
    names = Names(a.catalog, a.schema, a.volume)
    spark = _spark()

    batch_id, logs = _load_batch(spark, names, a.batch_id)
    scenario = str(logs["scenario"].iloc[0])
    llm, notes = None, []
    if a.agent_audit == "auto":
        getter = _secret_getter()
        found = find_llm(getter, a.secret_scope) if getter else None
        if found:
            try:
                llm = make_llm(found[0], found[1], a.llm_model or None)
                log.info("LLM key found for %s; running the agent team", found[0])
            except Exception as e:
                notes.append(f"Could not start the {found[0]} client: {type(e).__name__}")
    out = run_audit(logs, scenario=scenario, llm=llm)
    out.notes = notes + out.notes

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    report = render_report(out, batch_id, scenario, now)
    _write(spark, findings_rows(out, batch_id, scenario, now), names.table(FINDINGS), FINDINGS_DDL)
    _write(spark, alerts_rows(out, batch_id, scenario, now), names.table(ALERTS), ALERTS_DDL)
    _write(spark, [report_row(out, batch_id, scenario, now, report)], names.table(REPORTS), REPORTS_DDL)

    os.makedirs(names.reports_path, exist_ok=True)
    path = f"{names.reports_path}/audit_{batch_id}.md"
    with open(path, "w") as fh:
        fh.write(report)

    grant = _grant_app_access(spark, names, a.app_name)
    summary = {"batch_id": batch_id, "scenario": scenario, "mode": out.mode, "incidents": len(out.findings),
               "alerts": len(out.alerts), "score": out.score, "report": path, "app_access": grant}
    log.info("audit done: %s", json.dumps(summary))
    print(json.dumps(summary))
    if _truthy(a.fail_on_incident) and out.findings:
        raise SystemExit(f"{len(out.findings)} fairness incident(s) found in batch {batch_id}")
