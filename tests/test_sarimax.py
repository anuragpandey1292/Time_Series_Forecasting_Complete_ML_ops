"""Tests for SARIMAX with forecast-time-known exogenous variables."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

import forecasting.models.sarimax as sarimax_module
from forecasting.models import SARIMAXModel

EXOGENOUS_COLUMNS = ["onpromotion", "oil_price", "holiday_indicator", "payday"]


def make_train_data(days: int = 84) -> pd.DataFrame:
    """Build one daily sales series with varying exogenous inputs."""
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    index = np.arange(days)
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 44,
            "family": "GROCERY I",
            "sales": 45 + index * 0.2 + 3 * np.sin(index * 2 * np.pi / 7),
            "onpromotion": index % 4,
            "oil_price": 50 + np.sin(index / 5),
            "holiday_indicator": (index % 19 == 0).astype(int),
            "payday": (dates.day == 15) | (dates.day == dates.days_in_month),
        }
    )


def make_future_data(days: int = 16) -> pd.DataFrame:
    """Build future keys and exogenous values after the default training span."""
    dates = pd.date_range("2017-03-26", periods=days, freq="D")
    index = np.arange(days)
    return pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 44,
            "family": "GROCERY I",
            "onpromotion": index % 3,
            "oil_price": 51 + index / 10,
            "holiday_indicator": (index == 2).astype(int),
            "payday": (dates.day == 15) | (dates.day == dates.days_in_month),
        }
    )


def test_sarimax_fit_predict_accepts_future_exog_and_returns_schema() -> None:
    """Fit and forecast with the four required future regressors."""
    model = SARIMAXModel()
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
    assert SARIMAXModel.order == (1, 1, 1)
    assert SARIMAXModel.seasonal_order == (1, 0, 1, 7)


def test_sarimax_rejects_validation_sales_as_prediction_input() -> None:
    """Future actual sales cannot enter the prediction frame."""
    model = SARIMAXModel()
    model.fit(make_train_data())
    future = make_future_data(days=2)

    with pytest.raises(ValueError, match="must not contain sales"):
        model.predict(future.assign(sales=[100_000, 200_000]))


def test_sarimax_preserves_missing_historical_dates_without_sales_fill() -> None:
    """A missing date remains absent from statsmodels' observed input rows."""
    train = make_train_data().drop(index=20).reset_index(drop=True)
    model = SARIMAXModel()

    model.fit(train)

    labels = pd.DatetimeIndex(model._fitted_model.model.data.row_labels)
    assert model._fitted_model.nobs == len(make_train_data())
    assert pd.Timestamp("2017-01-21") in labels
    assert labels.freqstr == "D"
    assert np.isnan(model._fitted_model.model.endog[20, 0])


def test_sarimax_training_index_has_explicit_daily_frequency() -> None:
    """The fitted statsmodels model receives a DatetimeIndex with freq D."""
    model = SARIMAXModel()
    model.fit(make_train_data())

    fitted_index = model._fitted_model.model.data.row_labels
    assert isinstance(fitted_index, pd.DatetimeIndex)
    assert fitted_index.freqstr == "D"


def test_sarimax_future_exog_uses_daily_frequency_and_predicts_across_gap() -> None:
    """Forecast regressors get a daily index after a gapped training history."""
    train = make_train_data().drop(index=20).reset_index(drop=True)
    model = SARIMAXModel()
    model.fit(train)
    future = make_future_data(days=3)

    with patch.object(
        model._fitted_model, "forecast", wraps=model._fitted_model.forecast
    ) as forecast:
        predictions = model.predict(future)

    assert len(predictions) == 3
    forecast_exog = forecast.call_args.kwargs["exog"]
    assert forecast_exog.index.freqstr == "D"
    assert forecast_exog.index.equals(pd.date_range("2017-03-26", periods=3, freq="D"))


def test_sarimax_rejects_invalid_future_dates_and_series() -> None:
    """Future data must match the fitted series and immediately follow history."""
    model = SARIMAXModel()
    model.fit(make_train_data())

    with pytest.raises(ValueError, match="fitted store-family series"):
        model.predict(make_future_data(days=1).assign(store_nbr=1))

    irregular = make_future_data(days=2)
    irregular.loc[1, "date"] = pd.Timestamp("2017-03-28")
    with pytest.raises(ValueError, match="consecutive days"):
        model.predict(irregular)


def test_sarimax_validates_exogenous_columns_and_values() -> None:
    """Missing or non-finite regressors fail with a clear input error."""
    model = SARIMAXModel()
    with pytest.raises(ValueError, match="missing required columns"):
        model.fit(make_train_data().drop(columns="oil_price"))

    model.fit(make_train_data())
    invalid_future = make_future_data(days=1)
    invalid_future.loc[0, "oil_price"] = np.nan
    with pytest.raises(ValueError, match="exogenous values must be finite"):
        model.predict(invalid_future)


def test_sarimax_excludes_leading_incomplete_exogenous_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The leading oil NaN and its target are both excluded from fitting."""
    captured: dict[str, pd.DataFrame | pd.Series] = {}

    class CapturingSARIMAX:
        def __init__(self, endog, exog, **kwargs) -> None:
            captured["endog"] = endog.copy()
            captured["exog"] = exog.copy()

        def fit(
            self,
            *,
            method: str,
            maxiter: int,
            disp: bool,
        ) -> object:
            return self

    monkeypatch.setattr(sarimax_module, "SARIMAX", CapturingSARIMAX)
    train = make_train_data()
    train.loc[0, "oil_price"] = np.nan
    leading_sales = train.loc[0, "sales"]

    SARIMAXModel().fit(train)

    fitted_endog = captured["endog"]
    fitted_exog = captured["exog"]
    assert isinstance(fitted_endog, pd.Series)
    assert isinstance(fitted_exog, pd.DataFrame)
    assert len(fitted_endog) == len(train) - 1
    assert leading_sales not in fitted_endog.to_numpy()
    assert pd.Timestamp("2017-01-01") not in fitted_endog.index
    assert fitted_endog.index[0] == pd.Timestamp("2017-01-02")


def test_sarimax_uses_powell_optimizer_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fit uses Powell with the configured iteration limit and visible output."""
    fit_options: dict[str, object] = {}

    class CapturingSARIMAX:
        def __init__(self, endog, exog, **kwargs) -> None:
            pass

        def fit(
            self,
            *,
            method: str,
            maxiter: int,
            disp: bool,
        ) -> object:
            fit_options.update(method=method, maxiter=maxiter, disp=disp)
            return self

    monkeypatch.setattr(sarimax_module, "SARIMAX", CapturingSARIMAX)

    SARIMAXModel().fit(make_train_data())

    assert fit_options == {"method": "powell", "maxiter": 200, "disp": False}


def test_sarimax_rejects_internal_nonfinite_exogenous_values() -> None:
    """A non-finite regressor after training begins is never silently dropped."""
    train = make_train_data()
    train.loc[20, "oil_price"] = np.nan

    with pytest.raises(ValueError, match="after the first complete exogenous row"):
        SARIMAXModel().fit(train)


def test_sarimax_rejects_non_chronological_training_data() -> None:
    """Training order must be chronological rather than silently sorted."""
    train = make_train_data()
    train.loc[[1, 2], :] = train.loc[[2, 1], :].to_numpy()

    with pytest.raises(ValueError, match="chronological"):
        SARIMAXModel().fit(train)
