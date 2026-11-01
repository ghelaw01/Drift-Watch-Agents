"""Governance knowledge the Governance agent looks up: NIST AI RMF 1.0 subcategories and actions.

Subcategory text is paraphrased from the NIST AI RMF 1.0 core (NIST AI 100-1, 2023).
Check the official playbook before you rely on it: https://airc.nist.gov/airmf-resources/playbook/
"""
from __future__ import annotations

NIST_AI_RMF = {
    "MEASURE 2.4": "The AI system's functionality and behavior are monitored when in production.",
    "MEASURE 2.11": "Fairness and bias are evaluated and the results are documented.",
    "MANAGE 2.4": "Mechanisms exist, with assigned owners, to supersede, disengage, or deactivate an AI "
                  "system whose outcomes are inconsistent with its intended use.",
    "MANAGE 4.1": "Post-deployment monitoring plans are implemented, including incident response, "
                  "recovery, and change management.",
    "GOVERN 1.5": "Ongoing monitoring and periodic review of the risk management process are planned, "
                  "with roles and responsibilities defined.",
}

PLAYBOOK = {
    "data_quality": {
        "action": "fix_data_pipeline",
        "why": "The patients did not change; the data describing them did. Retraining on broken data "
               "would bake the error in. Fix the upstream feed, then backfill and re-score.",
        "controls": ["MEASURE 2.4", "MEASURE 2.11", "MANAGE 4.1"],
        "owner": "Data engineering, with the model owner",
        "interim": "Route affected patients to manual review until scores are corrected.",
    },
    "population_change": {
        "action": "recalibrate_or_retrain",
        "why": "The patients really changed, and the model is now under-predicting their risk. "
               "Recalibrate for the group, or retrain on recent data, then validate fairness again.",
        "controls": ["MEASURE 2.11", "MANAGE 2.4", "MANAGE 4.1"],
        "owner": "Model owner, with clinical leadership sign-off",
        "interim": "Lower the flagging threshold for the affected group, or add manual review.",
    },
    "unclear": {
        "action": "escalate_for_review",
        "why": "The evidence does not separate a data problem from a real change. A person must look.",
        "controls": ["MEASURE 2.11", "GOVERN 1.5"],
        "owner": "AI governance committee",
        "interim": "Keep monitoring weekly; consider manual review for the group.",
    },
}


def lookup(cause_type: str) -> dict:
    entry = PLAYBOOK.get(cause_type, PLAYBOOK["unclear"])
    return {**entry, "controls": {c: NIST_AI_RMF[c] for c in entry["controls"]}}
