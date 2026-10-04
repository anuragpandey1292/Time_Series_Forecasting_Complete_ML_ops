"""Shared interface and input checks for forecasting models."""

from __future__ import annotations

from typing import Protocol

import pandas as pd

SERIES_COLUMNS = ["store_nbr", "family"]
TRAIN_COLUMNS = ["date", *SERIES_COLUMNS, "sales"]
FUTURE_COLUMNS = ["date", *SERIES_COLUMNS]
PREDICTION_COLUMNS = ["date", *SERIES_COLUMNS, "prediction"]


class ForecastModel(Protocol):
    """Minimal fit/predict interface for forecasting implementations."""

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit model state using historical observations only."""
        ...

    def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Predict rows identified by their future dates and series keys."""
        ...


def prepare_train_data(train_data: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy the columns needed to fit a baseline model."""
    if not isinstance(train_data, pd.DataFrame):
        raise TypeError("train_data must be a pandas DataFrame")
    if train_data.empty:
        raise ValueError("train_data must not be empty")
    missing = sorted(set(TRAIN_COLUMNS) - set(train_data.columns))
    if missing:
        raise ValueError(f"train_data is missing required columns: {missing}")

    prepared = train_data.loc[:, TRAIN_COLUMNS].copy()
    if prepared[TRAIN_COLUMNS].isna().any().any():
        raise ValueError("train_data required columns must not contain missing values")
    try:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("train_data.date must contain valid dates") from exc
    if not pd.api.types.is_numeric_dtype(prepared["sales"]):
        raise ValueError("train_data.sales must be numeric")
    if prepared.duplicated(subset=["date", *SERIES_COLUMNS]).any():
        raise ValueError(
            "train_data contains duplicate date/store_nbr/family observations"
        )
    return prepared


def prepare_future_data(future_data: pd.DataFrame) -> pd.DataFrame:
    """Validate future rows and exclude actual target values from prediction."""
    if not isinstance(future_data, pd.DataFrame):
        raise TypeError("future_data must be a pandas DataFrame")
    if future_data.empty:
        raise ValueError("future_data must not be empty")
    missing = sorted(set(FUTURE_COLUMNS) - set(future_data.columns))
    if missing:
        raise ValueError(f"future_data is missing required columns: {missing}")
    if "sales" in future_data.columns:
        raise ValueError(
            "future_data must not contain sales; future actuals are evaluation-only"
        )

    prepared = future_data.loc[:, FUTURE_COLUMNS].copy()
    if prepared[FUTURE_COLUMNS].isna().any().any():
        raise ValueError("future_data required columns must not contain missing values")
    try:
        prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("future_data.date must contain valid dates") from exc
    if prepared.duplicated(subset=["date", *SERIES_COLUMNS]).any():
        raise ValueError(
            "future_data contains duplicate date/store_nbr/family rows"
        )
    return prepared
