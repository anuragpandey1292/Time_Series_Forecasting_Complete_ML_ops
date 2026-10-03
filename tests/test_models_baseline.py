"""Tests for the naive and seasonal naive forecasting baselines."""

import pandas as pd
import pytest

from forecasting.models import ForecastModel, NaiveModel, SeasonalNaiveModel


def make_train_data(days: int = 14) -> pd.DataFrame:
    """Build two store-family histories with distinguishable sales values."""
    rows = []
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    for offset, (store_nbr, family) in enumerate(
        [(1, "FOODS"), (2, "BEVERAGES")]
    ):
        for day_number, date in enumerate(dates, start=1):
            rows.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": offset * 100 + day_number,
                    "onpromotion": 0,
                }
            )
    return pd.DataFrame(rows)


def make_future_data(days: int = 2) -> pd.DataFrame:
    """Build future forecast rows for both series, without target values."""
    dates = pd.date_range("2017-01-15", periods=days, freq="D")
    return pd.DataFrame(
        [
            {"date": date, "store_nbr": store, "family": family}
            for date in dates
            for store, family in [(1, "FOODS"), (2, "BEVERAGES")]
        ]
    )


@pytest.mark.parametrize("model", [NaiveModel(), SeasonalNaiveModel()])
def test_models_satisfy_forecast_model_protocol(model: ForecastModel) -> None:
    """Both baselines implement the shared fit/predict interface."""
    assert callable(model.fit)
    assert callable(model.predict)


def test_naive_predicts_latest_value_per_store_family() -> None:
    """Naive repeats each series' latest historical sales for future dates."""
    model = NaiveModel()
    model.fit(make_train_data())

    predictions = model.predict(make_future_data())

    assert predictions.columns.tolist() == [
        "date",
        "store_nbr",
        "family",
        "prediction",
    ]
    assert len(predictions) == 4
    assert (
        predictions.groupby(["store_nbr", "family"])["prediction"]
        .nunique()
        .eq(1)
        .all()
    )
    assert predictions.loc[predictions["store_nbr"] == 1, "prediction"].eq(14).all()
    assert predictions.loc[predictions["store_nbr"] == 2, "prediction"].eq(114).all()


def test_seasonal_naive_uses_value_exactly_seven_days_earlier() -> None:
    """The first forecast uses each series' observed value from date minus 7."""
    model = SeasonalNaiveModel()
    model.fit(make_train_data())

    predictions = model.predict(make_future_data(days=1))

    assert predictions.columns.tolist() == [
        "date",
        "store_nbr",
        "family",
        "prediction",
    ]
    assert predictions.loc[predictions["store_nbr"] == 1, "prediction"].item() == 8
    assert predictions.loc[predictions["store_nbr"] == 2, "prediction"].item() == 108


def test_seasonal_naive_recurses_without_future_actual_sales() -> None:
    """Later seasonal lags use earlier predictions rather than actual targets."""
    train = make_train_data()
    future_with_actuals = make_future_data(days=16).assign(sales=999_999)
    model = SeasonalNaiveModel()
    model.fit(train)
    features = future_with_actuals.drop(columns="sales")

    predictions = model.predict(features)

    first_week = predictions[predictions["date"] <= pd.Timestamp("2017-01-21")]
    second_week = predictions[
        (predictions["date"] >= pd.Timestamp("2017-01-22"))
        & (predictions["date"] <= pd.Timestamp("2017-01-28"))
    ]
    first_week_values = first_week.loc[
        first_week["store_nbr"] == 1, "prediction"
    ].tolist()
    second_week_values = second_week.loc[
        second_week["store_nbr"] == 1, "prediction"
    ].tolist()
    assert first_week_values == list(range(8, 15))
    assert second_week_values == first_week_values
    with pytest.raises(ValueError, match="must not contain sales"):
        model.predict(future_with_actuals)


def test_naive_fit_uses_only_training_observations() -> None:
    """Validation actuals stay outside fit and cannot replace the last train value."""
    complete_data = make_train_data(days=16)
    historical = complete_data[complete_data["date"] <= "2017-01-14"]
    future = make_future_data(days=1)
    future["sales"] = 999_999
    model = NaiveModel()

    model.fit(historical)
    predictions = model.predict(future.drop(columns="sales"))

    assert predictions.loc[predictions["store_nbr"] == 1, "prediction"].item() == 14


@pytest.mark.parametrize("model", [NaiveModel(), SeasonalNaiveModel()])
def test_models_reject_missing_required_columns(model: ForecastModel) -> None:
    """Fit and predict report missing required columns clearly."""
    with pytest.raises(ValueError, match="missing required columns"):
        model.fit(make_train_data().drop(columns="sales"))

    model.fit(make_train_data())
    with pytest.raises(ValueError, match="missing required columns"):
        model.predict(pd.DataFrame({"date": ["2017-01-15"]}))


@pytest.mark.parametrize("model", [NaiveModel(), SeasonalNaiveModel()])
def test_models_reject_empty_training_data(model: ForecastModel) -> None:
    """Both baselines refuse to fit empty training frames."""
    with pytest.raises(ValueError, match="must not be empty"):
        model.fit(make_train_data().iloc[0:0])


@pytest.mark.parametrize("seasonal_period", [0, -7])
def test_seasonal_naive_rejects_invalid_period(seasonal_period: int) -> None:
    """Seasonal periods must be positive integers."""
    with pytest.raises(ValueError, match="greater than zero"):
        SeasonalNaiveModel(seasonal_period=seasonal_period)


def test_naive_requires_fit_before_prediction() -> None:
    """Prediction before fitting raises a clear state error."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        NaiveModel().predict(make_future_data(days=1))


def test_seasonal_naive_requires_fit_before_prediction() -> None:
    """Seasonal prediction before fitting raises a clear state error."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        SeasonalNaiveModel().predict(make_future_data(days=1))
