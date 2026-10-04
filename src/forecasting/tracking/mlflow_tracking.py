"""Optional MLflow tracking for model-evaluation results."""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pandas as pd

EXPERIMENT_NAME = "favorita-forecasting"
ARTIFACT_COLUMNS = [
    "model",
    "store_nbr",
    "family",
    "forecast_origin",
    "rmsle",
    "mae",
    "rmse",
    "status",
]


@dataclass(frozen=True)
class TrackingOutcome:
    """Describe whether an evaluation run was recorded successfully."""

    status: str
    run_id: str | None = None
    message: str | None = None


def _runtime_seconds(results: pd.DataFrame) -> float:
    """Sum per-fold fit and prediction time without repeated series rows."""
    fold_times = results.drop_duplicates("forecast_origin").copy()
    fit = pd.to_numeric(fold_times.get("fit_seconds"), errors="coerce").fillna(0)
    predict = pd.to_numeric(
        fold_times.get("prediction_seconds"), errors="coerce"
    ).fillna(0)
    return float((fit + predict).sum())


def _model_parameters(model_name: str, model: Any) -> dict[str, Any]:
    """Extract the fixed configuration and installed library version."""
    if model_name == "LightGBM":
        config = model._params
        return {
            "learning_rate": config["learning_rate"],
            "num_leaves": config["num_leaves"],
            "num_boost_round": model._num_boost_round,
            "seed": config["seed"],
            "thread_count": config["num_threads"],
            "lightgbm_version": version("lightgbm"),
        }
    if model_name == "CatBoost":
        config = model._estimator.get_params()
        return {
            "iterations": config["iterations"],
            "depth": config["depth"],
            "learning_rate": config["learning_rate"],
            "loss_function": config["loss_function"],
            "random_seed": config["random_seed"],
            "thread_count": config["thread_count"],
            "catboost_version": version("catboost"),
        }
    raise ValueError(f"Unsupported MLflow model name: {model_name}")


def _log_params(mlflow: Any, params: dict[str, Any]) -> None:
    """Log parameters as MLflow-compatible scalar strings."""
    mlflow.log_params(
        {
            key: json.dumps(value, default=str)
            if isinstance(value, (list, tuple, dict))
            else str(value)
            for key, value in params.items()
        }
    )


def log_evaluation_run(
    *,
    model_name: str,
    model: Any | None,
    results: pd.DataFrame,
    forecast_horizon: int,
    number_of_series: int,
    number_of_folds: int,
    forecast_origins: list[str],
    runtime_seconds: float | None = None,
    project_root: Path,
    enabled: bool = True,
    model_parameters: dict[str, Any] | None = None,
    tags: dict[str, str] | None = None,
    run_name: str | None = None,
    model_artifact_logger: Any | None = None,
) -> TrackingOutcome:
    """Log aggregate metrics and a compact per-fold CSV to local MLflow.

    Tracking setup or write failures are returned to the caller rather than
    raised, so model evaluation output remains usable when MLflow is offline.
    Raw training and validation data are never logged.
    """
    if not enabled:
        return TrackingOutcome(status="disabled")

    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        root = Path(project_root).resolve()
        database_path = root / "mlflow.db"
        artifact_root = root / "mlartifacts" / EXPERIMENT_NAME
        artifact_root.mkdir(parents=True, exist_ok=True)
        tracking_uri = f"sqlite:///{database_path.as_posix()}"
        artifact_location = artifact_root.as_uri()
        mlflow.set_tracking_uri(tracking_uri)

        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment_id = client.create_experiment(
                EXPERIMENT_NAME, artifact_location=artifact_location
            )
        else:
            experiment_id = experiment.experiment_id
        mlflow.set_experiment(EXPERIMENT_NAME)

        successful = results.loc[results["status"].eq("success")]
        failed_fits = int((~results["status"].eq("success")).sum())
        metric_values = {
            "mean_rmsle": float(successful["rmsle"].mean()),
            "median_rmsle": float(successful["rmsle"].median()),
            "mean_mae": float(successful["mae"].mean()),
            "mean_rmse": float(successful["rmse"].mean()),
            "failed_fits": failed_fits,
            "runtime_seconds": (
                _runtime_seconds(results)
                if runtime_seconds is None
                else float(runtime_seconds)
            ),
        }
        common_params = {
            "model_name": model_name,
            "forecast_horizon": forecast_horizon,
            "number_of_series": number_of_series,
            "number_of_folds": number_of_folds,
            "forecast_origins": forecast_origins,
            "evaluation": "rolling-origin",
            "recursive_forecasting": True,
        }
        if model is not None:
            common_params["feature_count"] = model.feature_count
        if model_parameters is None:
            model_parameters = _model_parameters(model_name, model)
        params = {**common_params, **model_parameters}
        run_tags = {
            "dataset": "Favorita",
            "evaluation": "rolling-origin",
            "horizon": str(forecast_horizon),
            "stage": "baseline",
            "model_family": "global_ml",
        }
        if tags:
            run_tags.update(tags)

        with mlflow.start_run(
            experiment_id=experiment_id,
            run_name=run_name or f"{model_name}-rolling-origin",
        ) as active_run:
            _log_params(mlflow, params)
            mlflow.set_tags(run_tags)
            for key, value in metric_values.items():
                mlflow.log_metric(key, value)

            if model_artifact_logger is not None:
                model_artifact_logger(mlflow)

            artifact_frame = results.loc[:, ARTIFACT_COLUMNS]
            with tempfile.TemporaryDirectory() as temporary_directory:
                artifact_path = Path(temporary_directory) / "per_fold_results.csv"
                artifact_frame.to_csv(artifact_path, index=False)
                mlflow.log_artifact(str(artifact_path), artifact_path="evaluation")
            return TrackingOutcome(status="logged", run_id=active_run.info.run_id)
    except Exception as exc:
        return TrackingOutcome(
            status="unavailable", message=f"{type(exc).__name__}: {exc}"
        )
