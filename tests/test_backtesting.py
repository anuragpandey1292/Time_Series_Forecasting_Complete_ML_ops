"""Tests for the model-agnostic rolling-origin framework."""

import pandas as pd
import pytest

from forecasting.backtesting import BacktestFold, generate_rolling_folds


def make_daily_data(days: int = 50) -> pd.DataFrame:
    """Create a small, date-indexed dataset entirely in memory."""
    dates = pd.date_range("2017-01-01", periods=days, freq="D")
    return pd.DataFrame(
        {
            "store_nbr": 1,
            "family": "FOODS",
            "sales": range(days),
            "onpromotion": 0,
        },
        index=dates,
    ).rename_axis("date")


def test_generates_single_valid_fold() -> None:
    """A valid origin produces the expected train and validation boundaries."""
    data = make_daily_data()
    origin = pd.Timestamp("2017-01-20")

    folds = generate_rolling_folds(data, [origin])

    assert len(folds) == 1
    fold = folds[0]
    assert isinstance(fold, BacktestFold)
    assert fold.forecast_origin == origin
    assert fold.train_start == pd.Timestamp("2017-01-01")
    assert fold.train_end == origin
    assert fold.validation_start == pd.Timestamp("2017-01-21")
    assert fold.validation_end == pd.Timestamp("2017-02-05")


def test_generates_multiple_rolling_origins() -> None:
    """Multiple increasing origins create independent rolling folds."""
    data = make_daily_data()
    origins = pd.to_datetime(["2017-01-16", "2017-01-20", "2017-01-24"])

    folds = generate_rolling_folds(data, origins)

    assert [fold.forecast_origin for fold in folds] == list(origins)
    assert all(fold.train_end == fold.forecast_origin for fold in folds)


def test_validation_has_requested_sixteen_day_horizon() -> None:
    """The default horizon is 16 distinct consecutive validation dates."""
    fold = generate_rolling_folds(
        make_daily_data(), [pd.Timestamp("2017-01-20")]
    )[0]

    assert len(fold.validation_features) == 16
    assert fold.validation_features.index.is_unique
    assert fold.validation_target.index.equals(fold.validation_features.index)
    assert fold.validation_features.index.equals(
        pd.date_range(fold.validation_start, periods=16, freq="D")
    )


def test_training_and_validation_dates_do_not_overlap() -> None:
    """Training ends at the origin and validation begins the following day."""
    fold = generate_rolling_folds(
        make_daily_data(), [pd.Timestamp("2017-01-20")]
    )[0]

    assert fold.train_data.index.max() == fold.forecast_origin
    assert fold.validation_features.index.min() > fold.train_data.index.max()
    assert not fold.validation_features.columns.isin(["sales"]).any()


def test_invalid_horizon_raises() -> None:
    """A zero horizon is rejected."""
    with pytest.raises(ValueError, match="greater than zero"):
        generate_rolling_folds(
            make_daily_data(), [pd.Timestamp("2017-01-20")], horizon=0
        )


def test_invalid_forecast_origin_raises() -> None:
    """An origin absent from the dataset is rejected."""
    with pytest.raises(ValueError, match="present in the dataset"):
        generate_rolling_folds(
            make_daily_data(), [pd.Timestamp("2016-12-31")]
        )


def test_incomplete_horizon_is_skipped() -> None:
    """Origins too close to the dataset end do not produce short folds."""
    folds = generate_rolling_folds(
        make_daily_data(days=25), [pd.Timestamp("2017-01-20")]
    )

    assert folds == []


def test_unordered_dates_raise() -> None:
    """A dataset with a descending date index is rejected."""
    data = make_daily_data().sort_index(ascending=False)

    with pytest.raises(ValueError, match="ascending order"):
        generate_rolling_folds(data, [pd.Timestamp("2017-01-20")])
