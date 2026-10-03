"""Tests for the single-series ARIMA forecasting model."""

import numpy as np
import pandas as pd
import pytest

import forecasting.models.arima as arima_module
from forecasting.models import ARIMAModel


def make_train_data(days: int = 60) -> pd.DataFrame:
    """Create a positive, regular synthetic daily series."""
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    values = 30 + np.arange(days) * 0.4 + np.sin(np.arange(days) * 2 * np.pi / 7)
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 44,
            "family": "GROCERY I",
            "sales": values,
        }
    )


def make_future_data(days: int = 16) -> pd.DataFrame:
    """Create consecutive future keys for the synthetic series."""
    dates = pd.date_range("2017-03-02", periods=days, freq="D")
    return pd.DataFrame(
        {"date": dates, "store_nbr": 44, "family": "GROCERY I"}
    )


def test_arima_fit_predict_and_output_schema() -> None:
    """ARIMA fits a small series and returns one prediction per future date."""
    model = ARIMAModel()
    model.fit(make_train_data())

    predictions = model.predict(make_future_data())

    assert len(predictions) == 16
    assert predictions.columns.tolist() == [
        "date",
        "store_nbr",
        "family",
        "prediction",
    ]
    assert predictions["prediction"].notna().all()


def test_arima_does_not_accept_validation_actuals_as_inputs() -> None:
    """Future sales are rejected from predict and are never used for fitting."""
    historical = make_train_data()
    future = make_future_data(days=2)
    model = ARIMAModel()
    model.fit(historical)

    predictions = model.predict(future)
    with pytest.raises(ValueError, match="must not contain sales"):
        model.predict(future.assign(sales=[1_000_000, 2_000_000]))

    assert len(predictions) == len(future)
    assert historical["sales"].iloc[-1] < 1_000_000


@pytest.mark.parametrize(
    "invalid_train",
    [
        pd.DataFrame(),
        make_train_data().drop(columns="sales"),
        pd.concat(
            [make_train_data(), make_train_data().assign(store_nbr=1)],
            ignore_index=True,
        ),
        pd.concat([make_train_data(), make_train_data().iloc[[0]]], ignore_index=True),
    ],
)
def test_arima_rejects_invalid_training_data(invalid_train: pd.DataFrame) -> None:
    """Empty, incomplete, mixed-series, and irregular data fail clearly."""
    with pytest.raises(ValueError):
        ARIMAModel().fit(invalid_train)


def test_arima_rejects_invalid_future_series() -> None:
    """Future keys must identify the fitted series and consecutive dates."""
    model = ARIMAModel()
    model.fit(make_train_data())

    with pytest.raises(ValueError, match="fitted store-family series"):
        model.predict(make_future_data(days=1).assign(store_nbr=1))
    irregular_future = make_future_data(days=2)
    irregular_future.loc[1, "date"] = pd.Timestamp("2017-03-04")
    with pytest.raises(ValueError, match="consecutive days"):
        model.predict(irregular_future)


def test_arima_preserves_missing_training_dates_without_filling_sales() -> None:
    """An absent historical date remains a missing observation in ARIMA input."""
    train = make_train_data().drop(index=10)
    model = ARIMAModel()

    model.fit(train)

    assert model._fitted_model.nobs == len(make_train_data())


def test_arima_fit_failure_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Statsmodels fitting failures are raised with ARIMA context."""

    class FailingARIMA:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def fit(self):
            raise ValueError("synthetic fitting failure")

    monkeypatch.setattr(arima_module, "ARIMA", FailingARIMA)

    with pytest.raises(RuntimeError, match="ARIMA.*fitting failed"):
        ARIMAModel().fit(make_train_data())


def test_arima_predict_requires_fit() -> None:
    """Prediction before model fitting raises a clear state error."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        ARIMAModel().predict(make_future_data(days=1))
