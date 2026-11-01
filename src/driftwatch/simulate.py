"""Simulated prediction logs from a deployed hospital readmission model, with planted drift.

Why simulate: to measure a drift detector you need ground truth, meaning you know exactly
which group drifted, in which month, and why. Real logs never tell you that.

The story: a model predicts 30-day readmission. It was trained on months 0-2. In some
scenarios, something changes for one group later (a coding change, a new patient
population, a label change). The model is not retrained, so its fairness drifts.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PAYERS = ["Medicare", "Medicaid", "Private", "Uninsured"]
PAYER_SHARES = [0.40, 0.25, 0.28, 0.07]
FEATURES = ["age", "prior_admissions", "length_of_stay", "num_medications", "has_followup"]


@dataclass
class Drift:
    """One planted change. Ground truth for the evaluation."""
    group: str                 # payer group affected
    start_month: int
    kind: str                  # "undercoding", "population_shift", "followup_loss"
    strength: float = 1.0
    cause_feature: str = ""    # the feature that really changed

    def __post_init__(self):
        if not self.cause_feature:
            self.cause_feature = {"undercoding": "prior_admissions",
                                  "population_shift": "age",
                                  "followup_loss": "has_followup"}[self.kind]


@dataclass
class Scenario:
    name: str
    months: int = 12
    patients_per_month: int = 4000
    drifts: list[Drift] = field(default_factory=list)
    seed: int = 0


def _patients(rng, n, month, drifts):
    payer = rng.choice(PAYERS, size=n, p=PAYER_SHARES)
    age = rng.normal(64, 14, n).clip(18, 98)
    age = np.where(payer == "Medicare", np.maximum(age, rng.normal(74, 7, n)), age)
    prior = rng.poisson(np.where(payer == "Medicaid", 1.6, 1.0))
    los = rng.gamma(2.0, 2.2, n)
    meds = rng.poisson(8 + (age - 60) / 10).clip(0)
    followup = rng.random(n) < np.where(payer == "Uninsured", 0.45, 0.75)

    # True readmission risk (the real world). It does not depend on the payer directly.
    logit = (-3.2 + 0.025 * (age - 60) + 0.55 * prior + 0.08 * los + 0.04 * meds - 0.7 * followup)
    true_prior = prior.copy()

    for d in drifts:
        if month < d.start_month:
            continue
        g = payer == d.group
        if d.kind == "undercoding":
            # A records change: prior admissions are no longer counted for this group.
            # The REAL risk is unchanged, but the model sees fewer prior admissions.
            prior = np.where(g, rng.binomial(true_prior, max(0.0, 1 - 0.75 * d.strength)), prior)
        elif d.kind == "population_shift":
            # A new, older, sicker population joins this group. Real risk rises with age.
            shift = 12 * d.strength
            age = np.where(g, (age + shift).clip(18, 98), age)
            logit = np.where(g, logit + 0.025 * shift + 0.35 * d.strength, logit)
        elif d.kind == "followup_loss":
            # A clinic closes: this group loses follow-up visits, and real risk rises.
            lost = g & followup & (rng.random(n) < 0.7 * d.strength)
            followup = np.where(lost, False, followup)
            logit = np.where(lost, logit + 0.7, logit)

    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame({"month": month, "payer": payer, "age": age.round(1), "prior_admissions": prior,
                         "length_of_stay": los.round(1), "num_medications": meds,
                         "has_followup": followup.astype(int), "readmitted": y.astype(int)})


def generate(scenario: Scenario) -> pd.DataFrame:
    """Prediction logs: one row per patient, with the deployed model's score and decision."""
    rng = np.random.default_rng(scenario.seed)
    frames = [_patients(rng, scenario.patients_per_month, m, scenario.drifts) for m in range(scenario.months)]
    df = pd.concat(frames, ignore_index=True)
    train = df[df["month"] < 3]
    model = LogisticRegression(max_iter=1000).fit(train[FEATURES], train["readmitted"])
    df["score"] = model.predict_proba(df[FEATURES])[:, 1].round(4)
    threshold = float(np.quantile(model.predict_proba(train[FEATURES])[:, 1], 0.8))   # flag the top 20%
    df["flagged"] = (df["score"] >= threshold).astype(int)
    return df


# A fixed suite of scenarios with known answers, used by the benchmark.
def benchmark_suite(replicas: int = 1) -> list[Scenario]:
    """13 scenarios with known answers. Each replica uses new random patients."""
    out = []
    for r in range(replicas):
        for sc in _base_suite():
            sc.seed += 1000 * r
            sc.name = sc.name if replicas == 1 else f"{sc.name}/r{r + 1}"
            out.append(sc)
    return out


def _base_suite() -> list[Scenario]:
    s = []
    for i, (group, kind, start, strength) in enumerate([
        ("Medicaid", "undercoding", 6, 1.0),
        ("Medicaid", "undercoding", 8, 0.6),
        ("Uninsured", "followup_loss", 5, 1.0),
        ("Medicare", "population_shift", 7, 1.0),
        ("Private", "undercoding", 4, 0.8),
        ("Uninsured", "undercoding", 6, 1.0),
        ("Medicaid", "followup_loss", 9, 0.8),
        ("Private", "population_shift", 5, 0.7),
    ]):
        s.append(Scenario(f"drift-{i + 1}", drifts=[Drift(group, start, kind, strength)], seed=100 + i))
    # Two problems at once, in two groups.
    s.append(Scenario("drift-9", drifts=[Drift("Medicaid", 6, "undercoding", 1.0),
                                         Drift("Medicare", 8, "population_shift", 1.0)], seed=109))
    for i in range(4):
        s.append(Scenario(f"no-drift-{i + 1}", seed=200 + i))
    return s


def expected_findings(scenario: Scenario) -> list[dict]:
    """Ground truth: which fairness incidents a correct audit should report.

    followup_loss is data drift, but the model sees has_followup, so its risk scores stay
    right and fairness does not degrade. A correct audit reports nothing for it.
    """
    out = []
    for d in scenario.drifts:
        if d.kind == "followup_loss":
            continue
        out.append({"group": d.group, "start_month": d.start_month, "cause_feature": d.cause_feature,
                    "cause_type": "data_quality" if d.kind == "undercoding" else "population_change",
                    "action": "fix_data_pipeline" if d.kind == "undercoding" else "recalibrate_or_retrain"})
    return out
