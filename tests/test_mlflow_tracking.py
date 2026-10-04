"""Tests for local and optional MLflow evaluation tracking."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from forecasting.tracking import EXPERIMENT_NAME, log_evaluation_run


class FakeCatBoost:
    """Minimal fitted-model metadata used to test tracking parameters."""

    feature_count = 12
    _estimator = SimpleNamespace(
        get_params=lambda: {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.05,
            "loss_function": "RMSE",
            "random_seed": 42,
            "thread_count": 8,
        }
    )


def make_results() -> pd.DataFrame:
    """Return small score-only rows without raw observations."""
    return pd.DataFrame(
        [
            {
                "model": "CatBoost",
                "store_nbr": 44,
                "family": "GROCERY I",
                "forecast_origin": "2017-06-28",
                "rmsle": 0.1,
                "mae": 2.0,
                "rmse": 3.0,
                "fit_seconds": 1.0,
                "prediction_seconds": 0.5,
                "status": "success",
                "error": None,
            },
            {
                "model": "CatBoost",
                "store_nbr": 1,
                "family": "PRODUCE",
                "forecast_origin": "2017-06-28",
                "rmsle": 0.3,
                "mae": 4.0,
                "rmse": 5.0,
                "fit_seconds": 1.0,
                "prediction_seconds": 0.5,
                "status": "success",
                "error": None,
            },
        ]
    )


def test_logs_metrics_parameters_tags_and_per_fold_artifact(tmp_path: Path) -> None:
    """A local SQLite run contains aggregate scores and only score artifacts."""
    outcome = log_evaluation_run(
        model_name="CatBoost",
        model=FakeCatBoost(),
        results=make_results(),
        forecast_horizon=16,
        number_of_series=2,
        number_of_folds=1,
        forecast_origins=["2017-06-28"],
        project_root=tmp_path,
    )

    assert outcome.status == "logged"
    assert outcome.run_id is not None
    tracking_uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    client = MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(outcome.run_id)
    assert run.data.params["model_name"] == "CatBoost"
    assert run.data.params["feature_count"] == "12"
    assert run.data.params["iterations"] == "100"
    assert run.data.metrics["mean_rmsle"] == pytest.approx(0.2)
    assert run.data.metrics["failed_fits"] == 0
    assert run.data.metrics["runtime_seconds"] == pytest.approx(1.5)
    assert run.data.tags["dataset"] == "Favorita"
    assert run.data.tags["evaluation"] == "rolling-origin"
    assert run.data.tags["model_family"] == "global_ml"

    artifacts = client.list_artifacts(outcome.run_id, "evaluation")
    assert [artifact.path for artifact in artifacts] == [
        "evaluation/per_fold_results.csv"
    ]
    local_artifact = client.download_artifacts(
        outcome.run_id, "evaluation/per_fold_results.csv"
    )
    artifact_frame = pd.read_csv(local_artifact)
    assert list(artifact_frame.columns) == [
        "model",
        "store_nbr",
        "family",
        "forecast_origin",
        "rmsle",
        "mae",
        "rmse",
        "status",
    ]
    assert "sales" not in artifact_frame.columns
    assert client.get_experiment_by_name(EXPERIMENT_NAME) is not None


def test_tracking_can_be_disabled(tmp_path: Path) -> None:
    """Disabling tracking has no side effects and returns normally."""
    outcome = log_evaluation_run(
        model_name="CatBoost",
        model=FakeCatBoost(),
        results=make_results(),
        forecast_horizon=16,
        number_of_series=2,
        number_of_folds=1,
        forecast_origins=["2017-06-28"],
        project_root=tmp_path,
        enabled=False,
    )

    assert outcome.status == "disabled"
    assert not (tmp_path / "mlflow.db").exists()


def test_classical_parameters_can_be_logged_without_a_model(tmp_path: Path) -> None:
    """Classical runs use the same helper with explicit model parameters."""
    outcome = log_evaluation_run(
        model_name="ARIMA",
        model=None,
        model_parameters={"order": "(1,1,1)"},
        results=make_results().assign(model="ARIMA"),
        forecast_horizon=16,
        number_of_series=2,
        number_of_folds=1,
        forecast_origins=["2017-06-28"],
        project_root=tmp_path,
        tags={"model_family": "classical"},
    )

    assert outcome.status == "logged"
    client = MlflowClient(
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    )
    run = client.get_run(outcome.run_id)
    assert run.data.params["order"] == "(1,1,1)"
    assert run.data.tags["model_family"] == "classical"


def test_tracking_unavailability_is_reported_not_raised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable MLflow package does not escape the tracking boundary."""
    monkeypatch.setitem(sys.modules, "mlflow", None)
    outcome = log_evaluation_run(
        model_name="CatBoost",
        model=FakeCatBoost(),
        results=make_results(),
        forecast_horizon=16,
        number_of_series=2,
        number_of_folds=1,
        forecast_origins=["2017-06-28"],
        project_root=tmp_path,
    )

    assert outcome.status == "unavailable"
    assert "ModuleNotFoundError" in (outcome.message or "")
