"""Train the readmission model, measure its fairness per payer, and log it to MLflow.

With window="baseline" this trains exactly like driftwatch.simulate.generate (months 0-2,
LogisticRegression, flag the top 20%), so the registered model is the deployed model whose
logs the audit checks. window="recent" retrains on the latest months instead. On undercoded
data that bakes the data error into the model, which is the trap DriftWatch warns about.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from driftwatch.simulate import FEATURES

from .config import BASELINE_MONTHS, TOP_SHARE_FLAGGED


@dataclass
class TrainResult:
    model: LogisticRegression
    threshold: float
    train_months: list[int]
    params: dict
    metrics: dict
    fairness: pd.DataFrame            # one row per (window, payer)
    input_example: pd.DataFrame = field(repr=False, default=None)


def _months(df: pd.DataFrame, window: str, baseline_months: int) -> list[int]:
    months = sorted(int(m) for m in df["month"].unique())
    if window == "baseline":
        return [m for m in months if m < baseline_months]
    if window == "recent":
        return months[-baseline_months:]
    raise ValueError("window must be 'baseline' or 'recent'")


def fairness_by_group(df: pd.DataFrame, score: np.ndarray, threshold: float, group_col: str = "payer") -> pd.DataFrame:
    """TPR, selection rate, and calibration gap per group, for the given scores."""
    d = df.assign(_score=score, _flag=(score >= threshold).astype(int))
    rows = []
    for g, part in d.groupby(group_col):
        pos = part[part["readmitted"] == 1]
        rows.append({"group": g, "n": len(part),
                     "tpr": float(pos["_flag"].mean()) if len(pos) else float("nan"),
                     "selection_rate": float(part["_flag"].mean()),
                     "calibration_gap": float(part["readmitted"].mean() - part["_score"].mean())})
    return pd.DataFrame(rows)


def train(df: pd.DataFrame, window: str = "baseline", baseline_months: int = BASELINE_MONTHS,
          top_share: float = TOP_SHARE_FLAGGED, group_col: str = "payer") -> TrainResult:
    train_months = _months(df, window, baseline_months)
    train_df = df[df["month"].isin(train_months)]
    recent_df = df[df["month"].isin(_months(df, "recent", baseline_months))]
    model = LogisticRegression(max_iter=1000).fit(train_df[FEATURES], train_df["readmitted"])
    train_score = model.predict_proba(train_df[FEATURES])[:, 1]
    threshold = float(np.quantile(train_score, 1 - top_share))
    recent_score = model.predict_proba(recent_df[FEATURES])[:, 1]

    fair = []
    for name, part, score in (("train", train_df, train_score), ("recent", recent_df, recent_score)):
        f = fairness_by_group(part, score, threshold, group_col)
        f.insert(0, "window", name)
        fair.append(f)
    fairness = pd.concat(fair, ignore_index=True)

    metrics = {
        "train_auc": float(roc_auc_score(train_df["readmitted"], train_score)),
        "recent_auc": float(roc_auc_score(recent_df["readmitted"], recent_score)),
        "threshold": threshold,
        "train_rows": float(len(train_df)),
    }
    for row in fairness.itertuples():
        key = f"{row.window}_{str(row.group).lower()}"
        metrics[f"{key}_tpr"] = row.tpr
        metrics[f"{key}_selection_rate"] = row.selection_rate
        metrics[f"{key}_calibration_gap"] = row.calibration_gap
    for name in ("train", "recent"):
        tpr = fairness.loc[fairness["window"] == name, "tpr"].dropna()
        metrics[f"{name}_max_tpr_gap"] = float(tpr.max() - tpr.min()) if len(tpr) else float("nan")
    metrics = {k: float(v) for k, v in metrics.items() if v == v}   # MLflow rejects NaN

    params = {"model": "LogisticRegression", "max_iter": 1000, "features": ",".join(FEATURES),
              "window": window, "train_months": ",".join(map(str, train_months)),
              "top_share_flagged": top_share, "group_col": group_col}
    return TrainResult(model, threshold, train_months, params, metrics, fairness,
                       input_example=train_df[FEATURES].head(5).reset_index(drop=True))


def should_promote(result: TrainResult, min_auc: float) -> tuple[bool, str]:
    """Promotion gate for the `champion` alias."""
    auc = result.metrics["recent_auc"]
    if auc < min_auc:
        return False, f"recent AUC {auc:.3f} is below the gate {min_auc:.3f}"
    return True, f"recent AUC {auc:.3f} passes the gate {min_auc:.3f}"


def log_to_mlflow(result: TrainResult, experiment: str | None = None, registered_model: str | None = None,
                  min_auc: float = 0.6, alias: str = "champion", tags: dict | None = None) -> dict:
    """Log params, metrics, the fairness table, and the model; register it and set the alias.

    registered_model is the 3-level Unity Catalog name (catalog.schema.model). If it is None,
    the model is logged but not registered (used by the local tests).
    """
    import mlflow
    from mlflow.models import infer_signature

    if experiment:
        mlflow.set_experiment(experiment)
    signature = infer_signature(result.input_example,
                                result.model.predict_proba(result.input_example))
    out = {}
    with mlflow.start_run(run_name=f"readmission-{result.params['window']}") as run:
        mlflow.set_tags({"project": "driftwatch", **(tags or {})})
        mlflow.log_params(result.params)
        mlflow.log_metrics(result.metrics)
        mlflow.log_table(data=result.fairness, artifact_file="fairness_by_payer.json")
        kwargs = dict(signature=signature, input_example=result.input_example,
                      pyfunc_predict_fn="predict_proba")
        try:   # MLflow 3 uses `name`; MLflow 2 uses `artifact_path`
            info = mlflow.sklearn.log_model(result.model, name="model", **kwargs)
        except TypeError:
            info = mlflow.sklearn.log_model(result.model, artifact_path="model", **kwargs)
        out.update(run_id=run.info.run_id, model_uri=info.model_uri)

        promote, why = should_promote(result, min_auc)
        mlflow.set_tag("promotion_decision", why)
        if registered_model:
            mlflow.set_registry_uri("databricks-uc")
            version = mlflow.register_model(info.model_uri, registered_model)
            out["version"] = str(version.version)
            client = mlflow.MlflowClient()
            client.set_model_version_tag(registered_model, version.version, "decision_threshold",
                                         f"{result.threshold:.6f}")
            if promote:
                client.set_registered_model_alias(registered_model, alias, version.version)
        out.update(promoted=bool(promote and registered_model), reason=why)
    return out
