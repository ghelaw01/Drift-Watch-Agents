# Lakeflow Declarative Pipeline (formerly Delta Live Tables) for DriftWatch.
#
#   bronze_prediction_logs   streaming table   raw Parquet files from the landing volume (Auto Loader)
#   silver_prediction_logs   streaming table   typed rows; data quality expectations drop bad rows
#   gold_fairness_metrics    materialized view per batch, month, and payer: TPR, calibration gap, ...
#   gold_fairness_watchlist  materialized view each month compared with the baseline months
#
# The pipeline's `root_path` (resources/pipeline.yml) is ../src, and Databricks adds it to
# sys.path, so the shared rules in driftwatch_databricks can be imported here.
# `spark` is provided by the pipeline runtime.
import dlt
from pyspark.sql import functions as F

from driftwatch_databricks import spark_gold
from driftwatch_databricks.config import (BRONZE, GOLD_METRICS, GOLD_WATCHLIST, LOG_COLUMNS, SILVER)
from driftwatch_databricks.quality import DROP_RULES, WARN_RULES

LANDING_PATH = spark.conf.get("driftwatch.landing_path")  # noqa: F821
BASELINE_MONTHS = int(spark.conf.get("driftwatch.baseline_months", "3"))  # noqa: F821


@dlt.table(
    name=BRONZE,
    comment="Raw prediction logs, one Parquet file per simulated batch, loaded with Auto Loader.",
    table_properties={"quality": "bronze"},
)
def bronze_prediction_logs():
    return (
        spark.readStream.format("cloudFiles")  # noqa: F821
        .option("cloudFiles.format", "parquet")
        .option("cloudFiles.schemaEvolutionMode", "rescue")
        .load(LANDING_PATH)
        .withColumn("_source_file", F.col("_metadata.file_path"))
        .withColumn("_ingested_at", F.current_timestamp())
    )


@dlt.table(
    name=SILVER,
    comment="Validated and typed prediction logs. Rows that break a DROP rule are removed.",
    table_properties={"quality": "silver"},
)
@dlt.expect_all_or_drop(DROP_RULES)
@dlt.expect_all(WARN_RULES)
def silver_prediction_logs():
    bronze = spark.readStream.table(BRONZE)  # noqa: F821
    typed = [F.col(c).cast(t).alias(c) for c, t in LOG_COLUMNS.items()]
    return bronze.select(*typed, "_source_file", "_ingested_at")


@dlt.table(
    name=GOLD_METRICS,
    comment="Fairness metrics per batch, month, and payer group (same math as driftwatch.analysis.group_metrics).",
    table_properties={"quality": "gold"},
)
def gold_fairness_metrics():
    return spark_gold.gold_metrics(spark.read.table(SILVER))  # noqa: F821


@dlt.table(
    name=GOLD_WATCHLIST,
    comment=("Each month compared with the baseline months. The flags are an effect-size screen only; "
             "the tested alerts are in fairness_alerts and fairness_findings (written by the audit job)."),
    table_properties={"quality": "gold"},
)
@dlt.expect("baseline_exists", "baseline_tpr IS NOT NULL")
def gold_fairness_watchlist():
    return spark_gold.watchlist(spark.read.table(SILVER), baseline_months=BASELINE_MONTHS)  # noqa: F821
