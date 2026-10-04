"""Univariate weekly-seasonal SARIMA forecasting for one store-family series."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.models.base import (
    PREDICTION_COLUMNS,
    SERIES_COLUMNS,
    prepare_future_data,
    prepare_train_data,
)


class SARIMAModel:
    """Fit one fixed-order statsmodels SARIMA model to one series."""

    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, 7)

    def __init__(self) -> None:
        """Create SARIMA(1,1,1)(1,0,1,7) forecaster."""
        self._fitted_model = None
        self._series_key: tuple[object, object] | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit the model using historical observations for one series only."""
        prepared = prepare_train_data(train_data).sort_values("date")
        series_keys = prepared[SERIES_COLUMNS].drop_duplicates()
        if len(series_keys) != 1:
            raise ValueError("SARIMAModel.fit requires exactly one store-family series")
        if not np.isfinite(prepared["sales"].to_numpy(dtype=float)).all():
            raise ValueError("train_data.sales must contain only finite values")
        if len(prepared) < 14:
            raise ValueError(
                "train_data must contain at least two seasonal periods "
                "(14 observations)"
            )

        dates = pd.DatetimeIndex(prepared["date"])
        if not (dates == dates.normalize()).all():
            raise ValueError("train_data.date must contain calendar-day dates")

        series_key = tuple(series_keys.iloc[0][SERIES_COLUMNS])
        # asfreq inserts missing timestamps with NaN, never fabricated sales.
        endog = prepared.set_index("date")["sales"].asfreq("D")
        try:
            fitted_model = SARIMAX(
                endog,
                order=self.order,
                seasonal_order=self.seasonal_order,
            ).fit(disp=False)
        except Exception as exc:
            raise RuntimeError(
                "SARIMA"
                f"{self.order}{self.seasonal_order} fitting failed for series "
                f"{series_key}: {exc}"
            ) from exc

        self._fitted_model = fitted_model
        self._series_key = series_key
        self._last_date = pd.Timestamp(dates[-1])

    def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast the consecutive requested dates without future actuals."""
        if self._fitted_model is None or self._last_date is None:
            raise RuntimeError("SARIMAModel must be fitted before predict()")
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
            raise RuntimeError(f"SARIMA forecasting failed: {exc}") from exc

        predictions = future.copy()
        predictions["prediction"] = np.asarray(forecast, dtype=float)
        return predictions.loc[:, PREDICTION_COLUMNS]
