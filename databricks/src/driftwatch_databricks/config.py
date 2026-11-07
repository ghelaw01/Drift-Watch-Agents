"""Names and constants shared by the pipeline, the jobs, and the app.

Kept free of third-party imports: the Lakeflow pipeline imports this module directly.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ---------------------------------------------------------------- tables
BRONZE = "bronze_prediction_logs"
SILVER = "silver_prediction_logs"
GOLD_METRICS = "gold_fairness_metrics"
GOLD_WATCHLIST = "gold_fairness_watchlist"
FINDINGS = "fairness_findings"
ALERTS = "fairness_alerts"
REPORTS = "audit_reports"

LANDING_SUBDIR = "prediction_logs"     # inside the landing volume
REPORTS_SUBDIR = "reports"             # inside the landing volume

# ---------------------------------------------------------------- statistics
# Same defaults as driftwatch.analysis.detect_drift. A test keeps them in sync.
BASELINE_MONTHS = 3
MIN_TPR_DROP = 0.10
MIN_CALIBRATION_SHIFT = 0.03
TOP_SHARE_FLAGGED = 0.20               # simulate.generate flags the top 20%

# ---------------------------------------------------------------- columns
KEY_COLUMNS = ["batch_id", "scenario", "month", "payer"]
LOG_COLUMNS = {
    "batch_id": "string",
    "scenario": "string",
    "generated_at": "timestamp",
    "month": "int",
    "payer": "string",
    "age": "double",
    "prior_admissions": "int",
    "length_of_stay": "double",
    "num_medications": "int",
    "has_followup": "int",
    "readmitted": "int",
    "score": "double",
    "flagged": "int",
}

# Spark DDL for the tables the audit job writes. Explicit, so an empty result still
# creates a table with the right columns.
FINDINGS_DDL = (
    "batch_id STRING, scenario STRING, audited_at TIMESTAMP, audit_mode STRING, "
    "`group` STRING, start_month INT, metrics STRING, cause_feature STRING, cause_type STRING, "
    "action STRING, owner STRING, interim STRING, controls STRING, nist_controls_json STRING, "
    "evidence STRING, matches_planted BOOLEAN"
)
ALERTS_DDL = (
    "batch_id STRING, scenario STRING, audited_at TIMESTAMP, `group` STRING, metric STRING, "
    "first_month INT, baseline DOUBLE, current DOUBLE, change DOUBLE, p_value DOUBLE"
)
REPORTS_DDL = (
    "batch_id STRING, scenario STRING, audited_at TIMESTAMP, audit_mode STRING, "
    "planted_incidents INT, found_incidents INT, false_incidents INT, notes STRING, report_md STRING"
)

_IDENT = re.compile(r"^[A-Za-z0-9_\-]+$")


def check_identifier(name: str) -> str:
    """Allow only simple catalog/schema/table names, so SQL built from them stays safe."""
    if not name or not _IDENT.match(name):
        raise ValueError(f"not a simple identifier: {name!r}")
    return name


@dataclass(frozen=True)
class Names:
    """Fully qualified Unity Catalog names for one catalog and schema."""
    catalog: str
    schema: str
    volume: str = "landing"

    def __post_init__(self):
        for part in (self.catalog, self.schema, self.volume):
            check_identifier(part)

    def table(self, name: str) -> str:
        return f"`{self.catalog}`.`{self.schema}`.`{check_identifier(name)}`"

    @property
    def volume_path(self) -> str:
        return f"/Volumes/{self.catalog}/{self.schema}/{self.volume}"

    @property
    def landing_path(self) -> str:
        return f"{self.volume_path}/{LANDING_SUBDIR}"

    @property
    def reports_path(self) -> str:
        return f"{self.volume_path}/{REPORTS_SUBDIR}"
