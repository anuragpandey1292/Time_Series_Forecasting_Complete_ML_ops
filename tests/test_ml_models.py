"""Focused tests for global LightGBM and CatBoost forecasting."""

import numpy as np
import pandas as pd
import pytest

from forecasting.models import CatBoostModel, LightGBMModel

MODEL_TYPES = [LightGBMModel, CatBoostModel]


def make_train_data(days: int = 430) -> pd.DataFrame:
    """Build two small uninterrupted store-family series."""
    dates = pd.date_range("2015-01-01", periods=days, freq="D")
    rows = []
    for store_nbr, family, offset in [
        (1, "GROCERY I", 20.0),
        (2, "BEVERAGES", 45.0),
    ]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "sales": offset + i * 0.1 + 4 * np.sin(i * 2 * np.pi / 7),
                    "onpromotion": i % 5,
                    "city": f"city-{store_nbr}",
                    "state": f"state-{store_nbr}",
                    "type": "A" if store_nbr == 1 else "B",
                    "cluster": store_nbr,
                    "holiday_indicator": int(i % 31 == 0),
                    "oil_price": 50.0 + i / 100,
                }
            )
    return pd.DataFrame(rows)


def make_future_data(train: pd.DataFrame, days: int = 16) -> pd.DataFrame:
    """Build known future covariates for both fitted series."""
    dates = pd.date_range(train["date"].max() + pd.Timedelta(days=1), periods=days)
    rows = []
    metadata = train.drop_duplicates("store_nbr").set_index("store_nbr")
    for store_nbr, family in [(1, "GROCERY I"), (2, "BEVERAGES")]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "store_nbr": store_nbr,
                    "family": family,
                    "onpromotion": i % 3,
                    "city": metadata.loc[store_nbr, "city"],
                    "state": metadata.loc[store_nbr, "state"],
                    "type": metadata.loc[store_nbr, "type"],
                    "cluster": metadata.loc[store_nbr, "cluster"],
                    "holiday_indicator": int(i == 5),
                    "oil_price": 51.0 + i / 100,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_global_model_fit_predict_and_categorical_handling(model_type) -> None:
    """Each global model predicts both series and uses native categories."""
    train = make_train_data()
    model = model_type()

    model.fit(train)
    predictions = model.predict(make_future_data(train))

    assert len(predictions) == 32
    assert predictions.columns.tolist() == [
        "date",
        "store_nbr",
        "family",
        "prediction",
    ]
    assert np.isfinite(predictions["prediction"]).all()
    assert predictions["prediction"].ge(0).all()
    assert set(model._categorical_columns) >= {
        "store_nbr",
        "family",
        "city",
        "state",
        "type",
        "cluster",
    }
    if model_type is LightGBMModel:
        assert model._estimator.pandas_categorical
    else:
        assert model._estimator.get_cat_feature_indices() == [
            model._feature_columns.index(name) for name in model._categorical_columns
        ]


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_future_actual_sales_are_rejected(model_type) -> None:
    """Validation targets cannot be supplied as recursive model inputs."""
    train = make_train_data()
    model = model_type()
    model.fit(train)

    with pytest.raises(ValueError, match="must not contain validation target"):
        model.predict(make_future_data(train).assign(sales=999_999.0))


def test_recursive_16_day_forecast_uses_earlier_predictions_for_lags() -> None:
    """Later recursive lags use generated predictions, not held-out targets."""
    train = make_train_data()
    model = LightGBMModel()
    model.fit(train)
    future = make_future_data(train).loc[lambda frame: frame["store_nbr"].eq(1)]

    class StepPredictor:
        def __init__(self) -> None:
            self.calls = 0
            self.lag_values: list[float] = []

        def predict(self, matrix: pd.DataFrame) -> np.ndarray:
            self.lag_values.append(float(matrix.iloc[0]["lag_1"]))
            value = 100.0 + self.calls
            self.calls += 1
            return np.array([value])

    fake = StepPredictor()
    model._estimator = fake

    predictions = model.predict(future)

    assert len(predictions) == 16
    assert fake.lag_values[1] == pytest.approx(100.0)
    assert fake.lag_values[2] == pytest.approx(101.0)
    assert predictions["prediction"].tolist() == list(np.arange(100.0, 116.0))


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_global_model_does_not_impute_missing_training_targets(model_type) -> None:
    """Callers must prepare missing targets explicitly before fitting."""
    train = make_train_data()
    train.loc[0, "sales"] = np.nan

    with pytest.raises(ValueError, match="prepare missing targets per fold"):
        model_type().fit(train)
