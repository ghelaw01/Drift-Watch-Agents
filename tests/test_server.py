import json

from mcp import Client

from driftwatch.server import build_server


async def _run(df):
    async with Client(build_server(df)) as client:
        names = sorted(t.name for t in (await client.list_tools()).tools)
        res = await client.call_tool("detect_fairness_drift", {})
    return names, json.loads(res.content[0].text)


def test_mcp_server_exposes_tools_and_detects(undercoded):
    import anyio
    names, out = anyio.run(_run, undercoded)
    assert names == ["data_overview", "detect_fairness_drift", "fairness_trend",
                     "find_root_cause", "lookup_governance"]
    assert out["alerts"][0]["group"] == "Medicaid"
