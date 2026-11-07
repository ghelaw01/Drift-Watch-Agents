"""DriftWatch on Databricks.

Pure-Python helpers for the Databricks deployment. Every module except `tasks` and
`spark_gold` works with pandas only, so it can be unit-tested without Spark.

This package is also put on `sys.path` of the Lakeflow pipeline (through the pipeline's
`root_path`), so importing it must stay cheap: no heavy imports at the top level.
"""

__version__ = "0.1.0"
