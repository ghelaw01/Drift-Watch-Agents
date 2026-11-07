"""DriftWatch app: fairness alerts, group trends, and NIST AI RMF controls.

Runs as a Databricks App. It reads the Delta tables written by the pipeline and the audit
job through the SQL warehouse. The app's own service principal is used to sign in.
"""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

import queries

CATALOG = os.getenv("DW_CATALOG", "workspace")
SCHEMA = os.getenv("DW_SCHEMA", "driftwatch")
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")

st.set_page_config(page_title="DriftWatch", layout="wide")


@st.cache_resource
def connection():
    cfg = Config()   # reads the app's service principal credentials from the environment
    return sql.connect(server_hostname=cfg.host, http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
                       credentials_provider=lambda: cfg.authenticate)


@st.cache_data(ttl=120)
def run(query: str, batch_id: str | None = None) -> pd.DataFrame:
    with connection().cursor() as cur:
        cur.execute(query, {"batch_id": batch_id} if batch_id else None)
        return cur.fetchall_arrow().to_pandas()


st.title("DriftWatch")
st.caption("Your model was fair at launch. Is it still? "
           f"Data: `{CATALOG}.{SCHEMA}` via SQL warehouse `{WAREHOUSE_ID or 'not set'}`.")

if not WAREHOUSE_ID:
    st.error("DATABRICKS_WAREHOUSE_ID is not set. Check the sql-warehouse resource of the app.")
    st.stop()

try:
    batches = run(queries.batches(CATALOG, SCHEMA))
except Exception as e:  # tables do not exist yet, or no permission
    st.warning("No audit results yet. Run the `driftwatch` job once, then refresh this page.")
    st.caption(f"Details: {type(e).__name__}: {e}")
    st.stop()

if batches.empty:
    st.info("The audit table is empty. Run the job, then refresh.")
    st.stop()

labels = {f"{r.batch_id}  ({r.scenario}, {r.audit_mode})": r.batch_id for r in batches.itertuples()}
batch_id = labels[st.sidebar.selectbox("Audited batch", list(labels))]
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()

findings = run(queries.findings(CATALOG, SCHEMA), batch_id)
alerts = run(queries.alerts(CATALOG, SCHEMA), batch_id)
trends = run(queries.trends(CATALOG, SCHEMA), batch_id)

c1, c2, c3 = st.columns(3)
c1.metric("Fairness incidents", len(findings))
c2.metric("Statistical alerts", len(alerts))
c3.metric("Groups checked", trends["group"].nunique() if not trends.empty else 0)

st.header("Incidents")
if findings.empty:
    st.success("No fairness drift found in this batch.")
for f in findings.itertuples():
    with st.container(border=True):
        st.subheader(f"{f.group}: from month {f.start_month}")
        st.write(f"**Cause:** {f.cause_type} via `{f.cause_feature}`. **Action:** `{f.action}`. "
                 f"**Owner:** {f.owner}.")
        if f.interim:
            st.write(f"**Until it is fixed:** {f.interim}")
        if f.evidence:
            st.caption(f"Evidence: {f.evidence}")
        controls = json.loads(f.nist_controls_json or "{}")
        if controls:
            st.table(pd.DataFrame({"NIST AI RMF control": list(controls), "What it asks": list(controls.values())}))

st.header("Group trends")
if not trends.empty:
    metric = st.radio("Metric", ["tpr", "calibration_gap"], horizontal=True,
                      format_func=lambda m: {"tpr": "True positive rate", "calibration_gap": "Calibration gap"}[m])
    st.line_chart(trends.pivot_table(index="month", columns="group", values=metric))
    st.caption("Months 0-2 are the baseline (training) period. Flags in the table below are an "
               "effect-size screen; the tested alerts are listed under 'Statistical alerts'.")
    st.dataframe(trends, hide_index=True, use_container_width=True)

st.header("Statistical alerts")
st.dataframe(alerts, hide_index=True, use_container_width=True)

st.header("Report")
rep = run(queries.report(CATALOG, SCHEMA), batch_id)
if not rep.empty:
    st.markdown(rep.iloc[0]["report_md"])
