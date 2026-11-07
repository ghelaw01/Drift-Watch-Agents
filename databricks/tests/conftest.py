"""Shared fixtures. Run from databricks/:  python -m pytest

The pyproject in this folder puts src/ and app/ on sys.path. The core `driftwatch`
package must be installed (pip install -e ..).
"""
import warnings

import pytest

from driftwatch_databricks.generate import make_batch

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def undercoded_batch():
    return make_batch("medicaid_undercoding", seed=11, batch_id="medicaid_undercoding-1")


@pytest.fixture(scope="session")
def clean_batch():
    return make_batch("none", seed=13, batch_id="none-1")


@pytest.fixture(scope="session")
def small_batch():
    """Small and fast, for shape and parity tests (too small for drift power)."""
    return make_batch("two_incidents", seed=5, batch_id="two_incidents-9", months=6, patients_per_month=400)
