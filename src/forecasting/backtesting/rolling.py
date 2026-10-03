"""Walk-forward fold generation for date-indexed forecasting data."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class BacktestFold:
    """One rolling-origin split with evaluation targets kept separate.

    ``train_data`` contains observations through the forecast origin.
    ``validation_features`` contains only future non-target columns, while
    ``validation_target`` is reserved for scoring predictions.
    """

    forecast_origin: pd.Timestamp
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_data: pd.DataFrame = field(repr=False, compare=False)
    validation_features: pd.DataFrame = field(repr=False, compare=False)
    validation_target: pd.Series = field(repr=False, compare=False)


def generate_rolling_folds(
    data: pd.DataFrame,
    forecast_origins: Iterable[pd.Timestamp | str],
    horizon: int = 16,
    target_column: str = "sales",
) -> list[BacktestFold]:
    """Generate complete rolling-origin folds from a DatetimeIndex frame.

    Rows at or before an origin form the training data. The following
    ``horizon`` calendar days form validation data. Validation target values
    are returned separately from validation features for evaluation only.
    Origins that lack a complete consecutive validation horizon are skipped.

    Args:
        data: Dataset indexed by dates. Multiple rows may share a date, for
            example one row per store-family series.
        forecast_origins: Increasing dates present in the dataset that mark
            the final day available for training.
        horizon: Number of consecutive calendar days to validate.
        target_column: Column reserved for the actual target values.

    Returns:
        Complete folds in forecast-origin order.

    Raises:
        TypeError: If the index is not a ``DatetimeIndex`` or horizon is not
            an integer.
        ValueError: If the horizon is not positive, dates or origins are not
            ordered, an origin is absent from the dataset, or the target
            column is missing.
    """
    if not isinstance(horizon, int) or isinstance(horizon, bool):
        raise TypeError("horizon must be an integer")
    if horizon <= 0:
        raise ValueError("horizon must be greater than zero")
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("data must have a pandas DatetimeIndex")
    if data.index.hasnans:
        raise ValueError("data index must not contain missing dates")
    if not data.index.is_monotonic_increasing:
        raise ValueError("data dates must be ordered in ascending order")
    if target_column not in data.columns:
        raise ValueError(f"target column {target_column!r} is missing from data")

    origins = [pd.Timestamp(origin) for origin in forecast_origins]
    if not origins:
        raise ValueError("at least one forecast origin is required")
    if origins != sorted(origins) or len(set(origins)) != len(origins):
        raise ValueError("forecast origins must be unique and in ascending order")

    available_dates = data.index.unique()
    available_date_set = set(available_dates)
    invalid_origins = [origin for origin in origins if origin not in available_date_set]
    if invalid_origins:
        raise ValueError(
            "forecast origins must be dates present in the dataset; invalid: "
            f"{invalid_origins}"
        )

    folds: list[BacktestFold] = []
    for origin in origins:
        validation_start = origin + pd.Timedelta(days=1)
        validation_end = origin + pd.Timedelta(days=horizon)
        expected_dates = pd.date_range(validation_start, validation_end, freq="D")
        present_dates = available_dates[
            (available_dates >= validation_start)
            & (available_dates <= validation_end)
        ]
        if not expected_dates.isin(present_dates).all():
            continue

        train_data = data.loc[:origin]
        validation_data = data.loc[validation_start:validation_end]
        validation_features = validation_data.drop(columns=[target_column]).copy()
        validation_target = validation_data[target_column].copy()

        folds.append(
            BacktestFold(
                forecast_origin=origin,
                train_start=pd.Timestamp(train_data.index.min()),
                train_end=pd.Timestamp(train_data.index.max()),
                validation_start=validation_start,
                validation_end=validation_end,
                train_data=train_data,
                validation_features=validation_features,
                validation_target=validation_target,
            )
        )

    return folds
