"""Tests for the fixed weekly-seasonal SARIMA model."""

import numpy as np
import pandas as pd
import pytest

from forecasting.models import SARIMAModel


def make_train_data(days: int = 70) -> pd.DataFrame:
    """Build one positive daily series with a weekly signal."""
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    values = 40 + np.arange(days) * 0.2 + 4 * np.sin(np.arange(days) * 2 * np.pi / 7)
    return pd.DataFrame(
        {"date": dates, "store_nbr": 44, "family": "GROCERY I", "sales": values}
    )


def make_future_data(days: int = 16) -> pd.DataFrame:
    """Build consecutive future keys following the default training fixture."""
    dates = pd.date_range("2017-03-12", periods=days, freq="D")
    return pd.DataFrame({"date": dates, "store_nbr": 44, "family": "GROCERY I"})


def test_sarima_fit_predict_and_output_schema() -> None:
    """SARIMA forecasts the requested horizon with the public output schema."""
    model = SARIMAModel()
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
    assert SARIMAModel.order == (1, 1, 1)
    assert SARIMAModel.seasonal_order == (1, 0, 1, 7)


def test_sarima_does_not_accept_validation_actuals_as_inputs() -> None:
    """Prediction rejects future target values rather than consuming them."""
    model = SARIMAModel()
    model.fit(make_train_data())
    future = make_future_data(days=2)

    prediction = model.predict(future)
    with pytest.raises(ValueError, match="must not contain sales"):
        model.predict(future.assign(sales=[1_000_000, 2_000_000]))

    assert len(prediction) == 2


def test_sarima_rejects_invalid_future_series() -> None:
    """Future requests must match the fitted series and be consecutive."""
    model = SARIMAModel()
    model.fit(make_train_data())

    with pytest.raises(ValueError, match="fitted store-family series"):
        model.predict(make_future_data(days=1).assign(store_nbr=1))

    irregular = make_future_data(days=2)
    irregular.loc[1, "date"] = pd.Timestamp("2017-03-14")
    with pytest.raises(ValueError, match="consecutive days"):
        model.predict(irregular)


def test_sarima_preserves_missing_historical_dates_without_filling_sales() -> None:
    """A missing date is represented by NaN in daily model endog."""
    train = make_train_data().drop(index=20)
    model = SARIMAModel()

    model.fit(train)

    assert model._fitted_model.nobs == len(make_train_data())
    assert np.isnan(model._fitted_model.model.endog[20, 0])
