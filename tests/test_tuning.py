"""Unit tests for LightGBMTuner and Bayesian hyperparameter optimization."""

import numpy as np
import pandas as pd
import pytest

from forecasting.tuning.optuna_tuner import FoldData, LightGBMTuner


def make_synthetic_fold(origin_date: str) -> FoldData:
    """Create a minimal synthetic fold for fast tuning unit tests."""
    origin = pd.Timestamp(origin_date)
    dates_train = pd.date_range(origin - pd.Timedelta(days=400), origin, freq="D")
    rows_train = []
    for store_nbr, family in [(1, "GROCERY I"), (2, "BEVERAGES")]:
        for i, date in enumerate(dates_train):
            rows_train.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": 50.0 + i * 0.05 + 5 * np.sin(i * 2 * np.pi / 7),
                    "onpromotion": i % 4,
                    "city": f"city-{store_nbr}",
                    "state": f"state-{store_nbr}",
                    "type": "A" if store_nbr == 1 else "B",
                    "cluster": store_nbr,
                    "holiday_indicator": int(i % 30 == 0),
                    "oil_price": 50.0,
                }
            )
    train_df = pd.DataFrame(rows_train)

    dates_val = pd.date_range(origin + pd.Timedelta(days=1), periods=16, freq="D")
    rows_future = []
    rows_actual = []
    for store_nbr, family in [(1, "GROCERY I"), (2, "BEVERAGES")]:
        for _i, date in enumerate(dates_val):
            rows_future.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "onpromotion": 1,
                    "city": f"city-{store_nbr}",
                    "state": f"state-{store_nbr}",
                    "type": "A" if store_nbr == 1 else "B",
                    "cluster": store_nbr,
                    "holiday_indicator": 0,
                    "oil_price": 50.0,
                }
            )
            rows_actual.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "actual": 60.0 + store_nbr * 10,
                }
            )
    future_df = pd.DataFrame(rows_future)
    actual_df = pd.DataFrame(rows_actual)
    return FoldData(
        forecast_origin=origin,
        train_features=train_df,
        future_features=future_df,
        actuals=actual_df,
    )


def test_tuner_rejects_empty_folds() -> None:
    """Tuner constructor must require at least one fold."""
    with pytest.raises(ValueError, match="folds must not be empty"):
        LightGBMTuner(folds=[])


def test_tuner_evaluate_configuration_returns_expected_metrics() -> None:
    """Evaluation helper returns complete metrics dictionary without NaNs."""
    fold = make_synthetic_fold("2017-06-28")
    tuner = LightGBMTuner(folds=[fold], seed=42)
    params = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 5,
        "seed": 42,
        "num_threads": 2,
        "verbosity": -1,
    }
    metrics = tuner.evaluate_configuration(params, num_boost_round=10)
    assert "mean_rmsle" in metrics
    assert "median_rmsle" in metrics
    assert "mean_mae" in metrics
    assert "mean_rmse" in metrics
    assert np.isfinite(metrics["mean_rmsle"])
    assert metrics["mean_rmsle"] >= 0.0


def test_tuner_optimize_runs_and_returns_best_result() -> None:
    """Optuna optimization executes requested trial count and returns valid result."""
    fold = make_synthetic_fold("2017-06-28")
    tuner = LightGBMTuner(folds=[fold], seed=42)
    result = tuner.optimize(n_trials=2, study_name="test-study")
    assert result.best_value >= 0.0
    assert result.best_num_boost_round > 0
    assert "learning_rate" in result.best_params
    assert "num_leaves" in result.best_params
    assert len(result.trials_dataframe) == 2
