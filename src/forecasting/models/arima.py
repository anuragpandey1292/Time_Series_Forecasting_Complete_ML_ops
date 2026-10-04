"""Univariate ARIMA forecasting model for one store-family series."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from forecasting.models.base import (
    PREDICTION_COLUMNS,
    SERIES_COLUMNS,
    prepare_future_data,
    prepare_train_data,
)


class ARIMAModel:
    """Fit one configurable statsmodels ARIMA model to a single series."""

    def __init__(self, order: Sequence[int] = (1, 1, 1)) -> None:
        """Create an ARIMA forecaster with an ``(p, d, q)`` order."""
        if len(order) != 3:
            raise ValueError("order must contain exactly three integers (p, d, q)")
        if any(
            not isinstance(value, int) or isinstance(value, bool) for value in order
        ):
            raise TypeError("ARIMA order values must be integers")
        if any(value < 0 for value in order):
            raise ValueError("ARIMA order values must be nonnegative")
        self.order = tuple(order)
        self._fitted_model = None
        self._series_key: tuple[object, object] | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit ARIMA using only the supplied history for one series."""
        prepared = prepare_train_data(train_data).sort_values("date")
        series_keys = prepared[SERIES_COLUMNS].drop_duplicates()
        if len(series_keys) != 1:
            raise ValueError("ARIMAModel.fit requires exactly one store-family series")
        if not np.isfinite(prepared["sales"].to_numpy(dtype=float)).all():
            raise ValueError("train_data.sales must contain only finite values")
        if len(prepared) <= sum(self.order):
            raise ValueError(
                "train_data has too few observations for the configured ARIMA order"
            )

        dates = pd.DatetimeIndex(prepared["date"])
        if not (dates == dates.normalize()).all():
            raise ValueError("train_data.date must contain calendar-day dates")

        series_key = tuple(series_keys.iloc[0][SERIES_COLUMNS])
        # Preserve absent dates as missing observations; do not fill sales.
        endog = prepared.set_index("date")["sales"].asfreq("D")
        try:
            fitted_model = ARIMA(
                endog,
                order=self.order,
            ).fit()
        except Exception as exc:
            raise RuntimeError(
                f"ARIMA{self.order} fitting failed for series {series_key}: {exc}"
            ) from exc

        self._fitted_model = fitted_model
        self._series_key = series_key
        self._last_date = pd.Timestamp(dates[-1])

    def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast the requested consecutive future dates for the fitted series."""
        if self._fitted_model is None or self._last_date is None:
            raise RuntimeError("ARIMAModel must be fitted before predict()")
        future = prepare_future_data(future_data)
        future_keys = future[SERIES_COLUMNS].drop_duplicates()
        requested_key = tuple(future_keys.iloc[0][SERIES_COLUMNS])
        if len(future_keys) != 1 or requested_key != self._series_key:
            raise ValueError(
                "future_data must contain only the fitted store-family series"
            )

        dates = pd.DatetimeIndex(future["date"])
        expected_dates = pd.date_range(
            self._last_date + pd.Timedelta(days=1), periods=len(future), freq="D"
        )
        if not dates.equals(expected_dates):
            raise ValueError(
                "future_data dates must be consecutive days immediately after training"
            )

        try:
            forecast = self._fitted_model.forecast(steps=len(future))
        except Exception as exc:
            raise RuntimeError(f"ARIMA{self.order} forecasting failed: {exc}") from exc

        predictions = future.copy()
        predictions["prediction"] = np.asarray(forecast, dtype=float)
        return predictions.loc[:, PREDICTION_COLUMNS]
