"""Tests for fit-failure handling in the SARIMA evaluator."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.backtesting import BacktestFold

EVALUATOR_PATH = Path(__file__).parents[1] / "scripts" / "evaluate_sarima.py"
EVALUATOR_SPEC = spec_from_file_location("evaluate_sarima", EVALUATOR_PATH)
assert EVALUATOR_SPEC is not None and EVALUATOR_SPEC.loader is not None
evaluator = module_from_spec(EVALUATOR_SPEC)
EVALUATOR_SPEC.loader.exec_module(evaluator)


def make_fold() -> BacktestFold:
    """Build one complete synthetic evaluation fold for all configured series."""
    origin = pd.Timestamp("2020-01-20")
    history_dates = pd.date_range("2020-01-01", origin)
    future_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=16)
    train_rows = []
    future_rows = []
    actual_values = []
    for store_nbr, family in evaluator.SERIES_CONFIG:
        train_rows.extend(
            {
                "date": date,
                "store_nbr": store_nbr,
                "family": family,
                "sales": float(index + 1),
            }
            for index, date in enumerate(history_dates)
        )
        future_rows.extend(
            {"date": date, "store_nbr": store_nbr, "family": family}
            for date in future_dates
        )
        actual_values.extend([2.0] * len(future_dates))

    train_data = pd.DataFrame(train_rows).set_index("date")
    validation_features = pd.DataFrame(future_rows).set_index("date")
    validation_target = pd.Series(
        actual_values,
        index=validation_features.index,
        name="sales",
    )
    return BacktestFold(
        forecast_origin=origin,
        train_start=history_dates.min(),
        train_end=origin,
        validation_start=future_dates.min(),
        validation_end=future_dates.max(),
        train_data=train_data,
        validation_features=validation_features,
        validation_target=validation_target,
    )


def test_evaluator_records_fit_failure_and_continues(monkeypatch, capsys) -> None:
    """A RuntimeError adds a failed row while other series are still scored."""

    class SelectivelyFailingSARIMA:
        def fit(self, train_data: pd.DataFrame) -> None:
            if train_data["store_nbr"].iloc[0] == 44:
                raise RuntimeError("synthetic LU decomposition failure")

        def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
            return future_data.assign(prediction=1.5)

    monkeypatch.setattr(evaluator.models, "SARIMAModel", SelectivelyFailingSARIMA)
    results = evaluator.evaluate_folds([make_fold()])

    assert len(results) == len(evaluator.SERIES_CONFIG)
    failed = results.loc[results["status"].eq("fit_failed")]
    successful = results.loc[results["status"].eq("success")]
    assert len(failed) == 1
    assert failed.iloc[0]["model"] == "SARIMA(1,1,1)(1,0,1,7)"
    assert failed.iloc[0]["error"] == "synthetic LU decomposition failure"
    assert failed[["rmsle", "mae", "rmse"]].isna().all().all()
    assert len(successful) == len(evaluator.SERIES_CONFIG) - 1

    evaluator.report_results(results)
    output = capsys.readouterr().out
    assert "Successful fits: 4" in output
    assert "Failed fits: 1" in output
    assert "synthetic LU decomposition failure" in output
    assert np.isfinite(successful["rmsle"]).all()
