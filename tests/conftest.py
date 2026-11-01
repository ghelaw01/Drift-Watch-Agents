import warnings

import pytest

from driftwatch.simulate import Drift, Scenario, generate

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def undercoded():
    return generate(Scenario("t-under", drifts=[Drift("Medicaid", 6, "undercoding")], seed=11))


@pytest.fixture(scope="session")
def older_population():
    return generate(Scenario("t-pop", drifts=[Drift("Medicare", 7, "population_shift")], seed=12))


@pytest.fixture(scope="session")
def clean():
    return generate(Scenario("t-clean", seed=13))
