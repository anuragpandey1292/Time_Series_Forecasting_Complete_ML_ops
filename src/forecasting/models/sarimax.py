"""SARIMAX forecasting for one store-family series with known exogenous data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from forecasting.data.imputation import impute_training_exogenous
from forecasting.models.base import (
    PREDICTION_COLUMNS,
    SERIES_COLUMNS,
    prepare_future_data,
    prepare_train_data,
)

EXOGENOUS_COLUMNS = ["onpromotion", "oil_price", "holiday_indicator", "payday"]


class SARIMAXModel:
    """Fit fixed weekly-seasonal SARIMAX to one store-family series."""

    order = (1, 1, 1)
    seasonal_order = (1, 0, 1, 7)

    def __init__(self) -> None:
        """Create SARIMAX(1,1,1)(1,0,1,7) with four fixed exogenous inputs."""
        self._fitted_model = None
        self._series_key: tuple[object, object] | None = None
        self._last_date: pd.Timestamp | None = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit on chronological observations and exogenous history only.

        Missing calendar dates remain absent; no target values are synthesized
        or imputed. A leading prefix with incomplete exogenous rows is excluded
        because no earlier known oil quote exists to construct those regressors;
        those rows' sales are excluded as well. Internal non-finite exogenous
        values are errors, not silently dropped.
        """
        if not isinstance(train_data, pd.DataFrame):
            raise TypeError("train_data must be a pandas DataFrame")
        required = {"date", *SERIES_COLUMNS, "sales", *EXOGENOUS_COLUMNS}
        missing = sorted(required - set(train_data.columns))
        if missing:
            raise ValueError(f"train_data is missing required columns: {missing}")
        validated = prepare_train_data(train_data)
        prepared = train_data.loc[
            :, ["date", *SERIES_COLUMNS, "sales", *EXOGENOUS_COLUMNS]
        ].copy()
        prepared["date"] = validated["date"].to_numpy()
        if prepared["date"].isna().any():
            raise ValueError("train_data.date must not contain missing values")
        if not prepared["date"].is_monotonic_increasing:
            raise ValueError("train_data must be in chronological date order")

        series_keys = prepared[SERIES_COLUMNS].drop_duplicates()
        if len(series_keys) != 1:
            raise ValueError(
                "SARIMAXModel.fit requires exactly one store-family series"
            )
        if not np.isfinite(prepared["sales"].to_numpy(dtype=float)).all():
            raise ValueError("train_data.sales must contain only finite values")
        if not all(
            pd.api.types.is_numeric_dtype(prepared[column])
            for column in EXOGENOUS_COLUMNS
        ):
            raise ValueError("train_data exogenous columns must be numeric")
        prepared[EXOGENOUS_COLUMNS] = prepared[EXOGENOUS_COLUMNS].astype(float)
        exog_values = prepared[EXOGENOUS_COLUMNS].to_numpy(dtype=float)
        complete_exog = np.isfinite(exog_values).all(axis=1)
        valid_positions = np.flatnonzero(complete_exog)
        if len(valid_positions) == 0:
            raise ValueError(
                "train_data has no row with complete finite exogenous values"
            )
        first_valid_position = int(valid_positions[0])
        internal_invalid = np.flatnonzero(~complete_exog[first_valid_position:])
        if len(internal_invalid):
            invalid_positions = internal_invalid + first_valid_position
            details = []
            for column_index, column in enumerate(EXOGENOUS_COLUMNS):
                column_invalid = ~np.isfinite(
                    exog_values[invalid_positions, column_index]
                )
                for position in invalid_positions[column_invalid]:
                    details.append(
                        f"{column} at {prepared.iloc[position]['date'].date()}"
                    )
            raise ValueError(
                "train_data contains non-finite exogenous values after the first "
                f"complete exogenous row: {', '.join(details)}"
            )
        if first_valid_position:
            prepared = prepared.iloc[first_valid_position:].copy()
        if len(prepared) < 14:
            raise ValueError(
                "train_data must contain at least two seasonal periods "
                "(14 complete-exogenous observations)"
            )

        dates = pd.DatetimeIndex(prepared["date"])
        if not (dates == dates.normalize()).all():
            raise ValueError("train_data.date must contain calendar-day dates")

        series_key = tuple(series_keys.iloc[0][SERIES_COLUMNS])
        daily_index = pd.date_range(dates.min(), dates.max(), freq="D")
        endog = prepared.set_index("date")["sales"].reindex(daily_index)
        exog_frame = prepared.set_index("date")[EXOGENOUS_COLUMNS].reindex(daily_index)
        exog_frame.index.name = "date"
        # Keep regressor gap handling centralized; never fill target values here.
        exog_filled = impute_training_exogenous(
            exog_frame.reset_index(),
            EXOGENOUS_COLUMNS,
            train_end=daily_index[-1],
        ).set_index("date")
        exog = exog_filled.loc[:, EXOGENOUS_COLUMNS].astype(float)
        exog.index = daily_index
        if not np.isfinite(exog.to_numpy(dtype=float)).all():
            raise ValueError("daily training exogenous values must be finite")
        try:
            fitted_model = SARIMAX(
                endog,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
            ).fit(method="powell", maxiter=200, disp=False)
        except Exception as exc:
            raise RuntimeError(
                "SARIMAX"
                f"{self.order}{self.seasonal_order} fitting failed for series "
                f"{series_key}: {exc}"
            ) from exc

        self._fitted_model = fitted_model
        self._series_key = series_key
        self._last_date = pd.Timestamp(dates[-1])

    def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast requested future keys using their known exogenous values."""
        if self._fitted_model is None or self._last_date is None:
            raise RuntimeError("SARIMAXModel must be fitted before predict()")
        required_exog = set(EXOGENOUS_COLUMNS)
        missing = sorted(required_exog - set(future_data.columns))
        if missing:
            raise ValueError(f"future_data is missing exogenous columns: {missing}")
        future = prepare_future_data(future_data)

        future_keys = future[SERIES_COLUMNS].drop_duplicates()
        requested_key = tuple(future_keys.iloc[0][SERIES_COLUMNS])
        if len(future_keys) != 1 or requested_key != self._series_key:
            raise ValueError(
                "future_data must contain only the fitted store-family series"
            )
        if not all(
            pd.api.types.is_numeric_dtype(future_data[column])
            for column in EXOGENOUS_COLUMNS
        ):
            raise ValueError("future_data exogenous columns must be numeric")
        exog = future_data.loc[:, EXOGENOUS_COLUMNS].reset_index(drop=True)
        if not np.isfinite(exog.to_numpy(dtype=float)).all():
            raise ValueError("future_data exogenous values must be finite")
        exog = exog.astype(float)

        dates = pd.DatetimeIndex(future["date"])
        if not (dates == dates.normalize()).all():
            raise ValueError("future_data.date must contain calendar-day dates")
        expected_dates = pd.date_range(
            self._last_date + pd.Timedelta(days=1), periods=len(future), freq="D"
        )
        if not dates.equals(expected_dates):
            raise ValueError(
                "future_data dates must be consecutive days immediately after training"
            )

        try:
            exog.index = expected_dates
            forecast = self._fitted_model.forecast(steps=len(future), exog=exog)
        except Exception as exc:
            raise RuntimeError(f"SARIMAX forecasting failed: {exc}") from exc

        predictions = future.copy()
        predictions["prediction"] = np.asarray(forecast, dtype=float)
        return predictions.loc[:, PREDICTION_COLUMNS]
