"""Focused tests for the unified training entry point."""

from __future__ import annotations

import sys

import pandas as pd
import pytest
import scripts.train as train


def test_model_factory_supports_all_cli_models() -> None:
    """Every documented model name maps to an existing implementation."""
    expected = {
        "naive": train.models.NaiveModel,
        "seasonal_naive": train.models.SeasonalNaiveModel,
        "arima": train.models.ARIMAModel,
        "sarima": train.models.SARIMAModel,
        "sarimax": train.models.SARIMAXModel,
        "lightgbm": train.models.LightGBMModel,
        "catboost": train.models.CatBoostModel,
    }
    for name, model_type in expected.items():
        assert isinstance(train.create_model(name), model_type)


def test_model_factory_rejects_unsupported_name() -> None:
    """Unsupported CLI values fail with a useful message."""
    with pytest.raises(ValueError, match="Unsupported model"):
        train.create_model("xgboost")


def test_model_specific_tracking_parameters() -> None:
    """The unified entry point preserves fixed model specifications."""
    assert train.model_parameters("naive")["strategy"] == "latest observed value"
    assert train.model_parameters("seasonal_naive")["seasonal_lag"] == 7
    assert train.model_parameters("arima")["order"] == (1, 1, 1)
    assert train.model_parameters("sarima")["seasonal_order"] == (1, 0, 1, 7)
    assert train.model_parameters("sarimax")["optimizer"] == "powell"


def test_cli_prints_pipeline_result_summary(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """CLI selection delegates to the pipeline and prints its result shape."""
    results = pd.DataFrame(
        [
            {
                "status": "success",
                "rmsle": 0.1,
                "mae": 2.0,
                "rmse": 3.0,
            }
        ]
    )
    outcome = train.tracking.TrackingOutcome(status="logged", run_id="run-123")

    def fake_run_pipeline(*args, **kwargs):
        return results, outcome, None

    monkeypatch.setattr(
        train,
        "run_pipeline",
        fake_run_pipeline,
    )
    monkeypatch.setattr(
        sys, "argv", ["train.py", "--model", "naive", "--mlflow-enabled", "false"]
    )

    train.main()

    output = capsys.readouterr().out
    assert "Model: naive" in output
    assert "Mean RMSLE: 0.1000" in output
    assert "MLflow run ID: run-123" in output
