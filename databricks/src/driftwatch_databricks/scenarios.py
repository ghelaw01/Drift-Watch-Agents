"""Named drift scenarios for the data generation job.

The job parameter `scenario` picks one of these, or any scenario from
driftwatch.simulate.benchmark_suite() by its name (for example "drift-3" or "no-drift-1").
"""
from __future__ import annotations

from driftwatch.simulate import Drift, Scenario, benchmark_suite

NAMED = {
    "none": [],
    "medicaid_undercoding": [Drift("Medicaid", 6, "undercoding")],
    "medicare_population_shift": [Drift("Medicare", 7, "population_shift")],
    "uninsured_followup_loss": [Drift("Uninsured", 5, "followup_loss")],
    "two_incidents": [Drift("Medicaid", 6, "undercoding"), Drift("Medicare", 8, "population_shift")],
}


def names() -> list[str]:
    return sorted(NAMED) + [s.name for s in benchmark_suite()]


def build(name: str, seed: int, months: int = 12, patients_per_month: int = 4000) -> Scenario:
    """Make the Scenario for a name. Unknown names raise ValueError with the valid choices."""
    if name in NAMED:
        drifts = [Drift(d.group, d.start_month, d.kind, d.strength) for d in NAMED[name]]
        return Scenario(name, months=months, patients_per_month=patients_per_month, drifts=drifts, seed=seed)
    for sc in benchmark_suite():
        if sc.name == name:
            sc.seed = seed
            sc.months = months
            sc.patients_per_month = patients_per_month
            return sc
    raise ValueError(f"unknown scenario {name!r}; choose one of: {', '.join(names())}")
