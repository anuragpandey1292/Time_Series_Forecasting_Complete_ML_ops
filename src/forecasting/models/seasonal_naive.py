"""Seasonal last-cycle baseline forecasting model."""

from __future__ import annotations

import pandas as pd

from forecasting.models.base import (
    PREDICTION_COLUMNS,
    SERIES_COLUMNS,
    prepare_future_data,
    prepare_train_data,
)


class SeasonalNaiveModel:
    """Use the value one seasonal period earlier, recursively if needed."""

    def __init__(self, seasonal_period: int = 7) -> None:
        """Create a seasonal naive forecaster with a positive day period."""
        if not isinstance(seasonal_period, int) or isinstance(seasonal_period, bool):
            raise TypeError("seasonal_period must be an integer")
        if seasonal_period <= 0:
            raise ValueError("seasonal_period must be greater than zero")
        self.seasonal_period = seasonal_period
        self._history: pd.DataFrame | None = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Retain recent historical values needed to seed seasonal forecasts."""
        prepared = prepare_train_data(train_data)
        latest_dates = prepared.groupby(SERIES_COLUMNS)["date"].transform("max")
        cutoff_dates = latest_dates - pd.Timedelta(days=self.seasonal_period - 1)
        self._history = prepared.loc[prepared["date"] >= cutoff_dates].copy()

    def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast each row from its exact seasonal lag without future actuals."""
        if self._history is None:
            raise RuntimeError("SeasonalNaiveModel must be fitted before predict()")
        future = prepare_future_data(future_data)

        # Keep a compact lookup of observed history and then add only predictions
        # made earlier in this same forecast, never future target observations.
        known_values = {
            (row.store_nbr, row.family, row.date): row.sales
            for row in self._history.itertuples(index=False)
        }
        ordered = future.sort_values([*SERIES_COLUMNS, "date"])
        predicted_values: dict[tuple[object, object, pd.Timestamp], object] = {}

        for row in ordered.itertuples(index=False):
            lag_date = row.date - pd.Timedelta(days=self.seasonal_period)
            lag_key = (row.store_nbr, row.family, lag_date)
            if lag_key not in known_values and lag_key not in predicted_values:
                raise ValueError(
                    "No observed or earlier predicted seasonal value for "
                    f"store_nbr={row.store_nbr!r}, family={row.family!r}, "
                    f"date={lag_date.date()}"
                )
            prediction = predicted_values.get(lag_key, known_values.get(lag_key))
            predicted_values[(row.store_nbr, row.family, row.date)] = prediction

        result = future.copy()
        result["prediction"] = [
            predicted_values[(row.store_nbr, row.family, row.date)]
            for row in result.itertuples(index=False)
        ]
        return result.loc[:, PREDICTION_COLUMNS]
