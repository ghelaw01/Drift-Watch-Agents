"""Serve the audit tools over MCP, so any MCP client (Claude Desktop, Cursor, your agent) can audit logs."""
from __future__ import annotations

import json

import pandas as pd
from mcp.server.mcpserver import MCPServer

from .tools import Toolbox


def build_server(logs: pd.DataFrame, baseline_months: int = 3, group_col: str = "payer") -> MCPServer:
    box = Toolbox(logs, baseline_months, group_col)
    mcp = MCPServer("driftwatch")

    @mcp.tool()
    def data_overview() -> str:
        """What the prediction logs contain: months, groups, and patients per group."""
        return box.call("data_overview", {})

    @mcp.tool()
    def detect_fairness_drift() -> str:
        """Run statistical drift tests on per-group fairness metrics (TPR and calibration)."""
        return box.call("detect_fairness_drift", {})

    @mcp.tool()
    def fairness_trend(group: str) -> str:
        """Month-by-month TPR, calibration gap, and selection rate for one group."""
        return box.call("fairness_trend", {"group": group})

    @mcp.tool()
    def find_root_cause(group: str, start_month: int) -> str:
        """Rank the inputs that shifted for one group since a month, and show whether the real outcome changed."""
        return box.call("find_root_cause", {"group": group, "start_month": start_month})

    @mcp.tool()
    def lookup_governance(cause_type: str) -> str:
        """Recommended action and NIST AI RMF controls for data_quality, population_change, or unclear."""
        return box.call("lookup_governance", {"cause_type": cause_type})

    return mcp


def load_logs(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    missing = {"month", "readmitted", "score", "flagged"} - set(df.columns)
    if missing:
        raise SystemExit(f"logs are missing columns: {sorted(missing)}")
    return df


__all__ = ["build_server", "load_logs", "json"]
