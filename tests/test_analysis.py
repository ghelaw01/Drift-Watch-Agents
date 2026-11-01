from driftwatch.analysis import detect_drift, psi, root_cause
from driftwatch.simulate import Drift, Scenario, expected_findings
import numpy as np


def test_undercoding_is_detected_at_the_right_month(undercoded):
    alerts = detect_drift(undercoded)
    assert {a.group for a in alerts} == {"Medicaid"}
    assert min(a.first_month for a in alerts) in (5, 6, 7)   # within one month


def test_no_false_alarm_on_clean_logs(clean):
    assert detect_drift(clean) == []


def test_root_cause_separates_bad_data_from_real_change(undercoded, older_population):
    under = root_cause(undercoded, "Medicaid", 6)
    assert under[0]["feature"] == "prior_admissions"
    assert abs(under[-1]["group_specific_outcome_change"]) < 0.02      # patients did not change
    pop = root_cause(older_population, "Medicare", 7)
    assert pop[0]["feature"] == "age"
    assert pop[-1]["group_specific_outcome_change"] >= 0.02            # patients really changed


def test_psi_is_zero_for_same_data_and_large_for_shift():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert psi(x, x) < 0.01
    assert psi(x, x + 1.5) > 0.2


def test_followup_loss_is_not_a_fairness_incident():
    assert expected_findings(Scenario("x", drifts=[Drift("Uninsured", 5, "followup_loss")])) == []
