"""The app's SQL builders (the Streamlit page itself needs a running warehouse)."""
import pytest

import queries


def test_queries_use_safe_names_and_a_bound_batch_id():
    for build in (queries.findings, queries.alerts, queries.trends, queries.report):
        q = build("workspace", "dev_me_driftwatch")
        assert "`workspace`.`dev_me_driftwatch`." in q
        assert ":batch_id" in q
    assert "LIMIT 20" in queries.batches("workspace", "driftwatch")


@pytest.mark.parametrize("bad", ["", "x; DROP TABLE t", "a`b", "a.b", "a b"])
def test_unsafe_names_are_rejected(bad):
    with pytest.raises(ValueError):
        queries.findings("workspace", bad)


def test_tables_used_by_the_app_are_the_ones_the_jobs_write():
    from driftwatch_databricks import config

    text = " ".join(f("c", "s") for f in (queries.batches, queries.findings, queries.alerts, queries.trends))
    for name in (config.REPORTS, config.FINDINGS, config.ALERTS, config.GOLD_WATCHLIST):
        assert f"`{name}`" in text
