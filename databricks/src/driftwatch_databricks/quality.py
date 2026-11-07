"""Data quality rules for bronze -> silver.

The same rules exist twice, on purpose:
  - as SQL expressions, used by the Lakeflow pipeline as expectations;
  - as pandas checks, used by unit tests and by anyone running locally.
A Spark test checks that both versions agree row by row.

DROP rules remove a row from silver (the audit must not see broken rows).
WARN rules keep the row but count the problem in the pipeline's data quality metrics.
"""
from __future__ import annotations

# name -> SQL boolean expression (true = the row is OK)
DROP_RULES: dict[str, str] = {
    "keys_present": "batch_id IS NOT NULL AND month IS NOT NULL AND payer IS NOT NULL",
    "month_not_negative": "month >= 0",
    "score_is_probability": "score IS NOT NULL AND score >= 0 AND score <= 1",
    "readmitted_is_binary": "readmitted IN (0, 1)",
    "flagged_is_binary": "flagged IN (0, 1)",
}

WARN_RULES: dict[str, str] = {
    "age_in_range": "age >= 0 AND age <= 120",
    "length_of_stay_not_negative": "length_of_stay >= 0",
    "counts_not_negative": "prior_admissions >= 0 AND num_medications >= 0",
    "followup_is_binary": "has_followup IN (0, 1)",
}


def _pandas_rules():
    """The same rules as pandas functions: frame -> boolean Series (True = OK).

    SQL treats NULL as "not OK" for a rule. pandas comparisons with NaN return False,
    which gives the same answer.
    """
    import pandas as pd  # noqa: F401  (only to make the dependency explicit)

    def isin(col, values):
        return lambda d: d[col].isin(values) & d[col].notna()

    return {
        "keys_present": lambda d: d["batch_id"].notna() & d["month"].notna() & d["payer"].notna(),
        "month_not_negative": lambda d: d["month"] >= 0,
        "score_is_probability": lambda d: d["score"].notna() & (d["score"] >= 0) & (d["score"] <= 1),
        "readmitted_is_binary": isin("readmitted", [0, 1]),
        "flagged_is_binary": isin("flagged", [0, 1]),
        "age_in_range": lambda d: (d["age"] >= 0) & (d["age"] <= 120),
        "length_of_stay_not_negative": lambda d: d["length_of_stay"] >= 0,
        "counts_not_negative": lambda d: (d["prior_admissions"] >= 0) & (d["num_medications"] >= 0),
        "followup_is_binary": isin("has_followup", [0, 1]),
    }


def check(df):
    """Return {rule name: boolean Series} for every rule (True = the row passes)."""
    rules = _pandas_rules()
    missing = set(DROP_RULES) | set(WARN_RULES)
    missing -= set(rules)
    if missing:  # pragma: no cover - guarded by a unit test
        raise RuntimeError(f"pandas version missing for rules: {sorted(missing)}")
    return {name: rules[name](df).fillna(False).astype(bool) for name in (*DROP_RULES, *WARN_RULES)}


def to_silver(df):
    """pandas version of the silver step: drop rows that fail a DROP rule.

    Returns (clean frame, {rule: number of failing rows}).
    """
    results = check(df)
    failures = {name: int((~ok).sum()) for name, ok in results.items()}
    keep = None
    for name in DROP_RULES:
        keep = results[name] if keep is None else keep & results[name]
    return df[keep].reset_index(drop=True), failures
