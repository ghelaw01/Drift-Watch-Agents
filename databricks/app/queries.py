"""SQL used by the DriftWatch app. Pure functions, so they are unit-tested without a warehouse.

Catalog and schema come from environment variables set by the bundle, never from the
viewer, and are checked against a strict pattern before they go into SQL. The batch id is
passed as a named query parameter.
"""
from __future__ import annotations

import re

_IDENT = re.compile(r"^[A-Za-z0-9_\-]+$")


def table(catalog: str, schema: str, name: str) -> str:
    for part in (catalog, schema, name):
        if not part or not _IDENT.match(part):
            raise ValueError(f"not a simple identifier: {part!r}")
    return f"`{catalog}`.`{schema}`.`{name}`"


def batches(catalog: str, schema: str, limit: int = 20) -> str:
    return (f"SELECT batch_id, scenario, max(audited_at) AS audited_at, first(audit_mode) AS audit_mode, "
            f"first(found_incidents) AS found_incidents, first(planted_incidents) AS planted_incidents "
            f"FROM {table(catalog, schema, 'audit_reports')} GROUP BY batch_id, scenario "
            f"ORDER BY audited_at DESC LIMIT {int(limit)}")


def findings(catalog: str, schema: str) -> str:
    return (f"SELECT `group`, start_month, metrics, cause_type, cause_feature, action, owner, interim, controls, "
            f"nist_controls_json, evidence, matches_planted, audit_mode "
            f"FROM {table(catalog, schema, 'fairness_findings')} WHERE batch_id = :batch_id "
            f"ORDER BY start_month, `group`")


def alerts(catalog: str, schema: str) -> str:
    return (f"SELECT `group`, metric, first_month, baseline, current, change, p_value "
            f"FROM {table(catalog, schema, 'fairness_alerts')} WHERE batch_id = :batch_id "
            f"ORDER BY first_month, `group`")


def trends(catalog: str, schema: str) -> str:
    return (f"SELECT month, `group`, n, tpr, baseline_tpr, tpr_change, calibration_gap, calibration_change, "
            f"tpr_drop_flag, calibration_flag FROM {table(catalog, schema, 'gold_fairness_watchlist')} "
            f"WHERE batch_id = :batch_id ORDER BY `group`, month")


def report(catalog: str, schema: str) -> str:
    return (f"SELECT report_md, notes FROM {table(catalog, schema, 'audit_reports')} "
            f"WHERE batch_id = :batch_id ORDER BY audited_at DESC LIMIT 1")
