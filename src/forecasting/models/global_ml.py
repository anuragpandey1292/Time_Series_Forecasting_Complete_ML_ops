"""Shared global tree-regression and recursive forecast behavior."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from forecasting.features.lags import LAG_PERIODS
from forecasting.features.pipeline import build_features
from forecasting.models.base import PREDICTION_COLUMNS, SERIES_COLUMNS

GRAIN_COLUMNS = ["date", *SERIES_COLUMNS]
AUDIT_COLUMNS = {"sales_original", "sales_imputation_method", "is_imputed"}
CATEGORICAL_HINTS = {"store_nbr", "family", "city", "state", "type", "cluster"}


class GlobalRecursiveRegressor:
    """Common fit and autoregressive rollout for global tree regressors."""

    def __init__(self, estimator: Any | None, categorical_mode: str) -> None:
        """Initialize an estimator adapter with backend categorical behavior."""
        self._estimator = estimator
        self._categorical_mode = categorical_mode
        self._feature_columns: list[str] = []
        self._covariate_columns: list[str] = []
        self._categorical_columns: list[str] = []
        self._category_levels: dict[str, list[str]] = {}
        self._history: pd.DataFrame | None = None
        self._series_keys: set[tuple[object, object]] = set()
        self._forecast_origin: pd.Timestamp | None = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Fit one global estimator on historical rows from all series.

        Callers must pass only the current fold's training data. Missing target
        observations must first be handled by the explicit modeling-safe data
        preparation layer; this method never imputes target values.
        """
        required = {*GRAIN_COLUMNS, "sales", "onpromotion"}
        missing = sorted(required - set(train_data.columns))
        if missing:
            raise ValueError(f"train_data is missing required columns: {missing}")
        if train_data.empty:
            raise ValueError("train_data must not be empty")
        if {"actual", "prediction"}.intersection(train_data.columns):
            raise ValueError("train_data contains a reserved target/leakage column")

        prepared = train_data.copy()
        try:
            prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError("train_data.date must contain valid dates") from exc
        if prepared["date"].isna().any():
            raise ValueError("train_data.date must not contain missing values")
        if prepared[[*SERIES_COLUMNS]].isna().any().any():
            raise ValueError("train_data series keys must not contain missing values")
        if prepared.duplicated(GRAIN_COLUMNS).any():
            raise ValueError("train_data must be unique on date + store + family")
        if not pd.api.types.is_numeric_dtype(prepared["sales"]):
            raise ValueError("train_data.sales must be numeric")
        sales = prepared["sales"].to_numpy(dtype=float)
        if not np.isfinite(sales).all():
            raise ValueError(
                "train_data.sales must be finite; prepare missing targets per fold"
            )
        if (sales < 0).any():
            raise ValueError("train_data.sales must be non-negative")
        if not pd.api.types.is_numeric_dtype(prepared["onpromotion"]):
            raise ValueError("train_data.onpromotion must be numeric")

        prepared = prepared.sort_values(GRAIN_COLUMNS, kind="stable").reset_index(
            drop=True
        )
        self._covariate_columns = [
            column
            for column in prepared.columns
            if column not in {*GRAIN_COLUMNS, "sales", *AUDIT_COLUMNS}
        ]
        if "onpromotion" not in self._covariate_columns:
            raise ValueError("train_data must include onpromotion")

        feature_frame = build_features(prepared)
        excluded = {"date", "sales", *AUDIT_COLUMNS}
        self._feature_columns = [
            column for column in feature_frame.columns if column not in excluded
        ]
        if not self._feature_columns:
            raise ValueError("train_data produced no model features")
        self._categorical_columns = [
            column
            for column in self._feature_columns
            if column in CATEGORICAL_HINTS
            or not pd.api.types.is_numeric_dtype(feature_frame[column])
        ]
        self._category_levels = {
            column: sorted(feature_frame[column].dropna().astype(str).unique())
            for column in self._categorical_columns
        }
        matrix = self._encode_features(feature_frame)
        self._fit_estimator(matrix, feature_frame["sales"].to_numpy(dtype=float))

        self._forecast_origin = pd.Timestamp(prepared["date"].max())
        cutoff = self._forecast_origin - pd.Timedelta(days=max(LAG_PERIODS))
        context_columns = [*GRAIN_COLUMNS, "sales", *self._covariate_columns]
        self._history = prepared.loc[
            prepared["date"].ge(cutoff), context_columns
        ].copy()
        self._series_keys = set(
            map(tuple, prepared.loc[:, SERIES_COLUMNS].drop_duplicates().to_numpy())
        )

    def _encode_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a backend-ready matrix with fitted categorical vocabularies."""
        matrix = frame.loc[:, self._feature_columns].copy()
        for column in self._categorical_columns:
            values = matrix[column].astype("string")
            if self._categorical_mode == "lightgbm":
                matrix[column] = pd.Categorical(
                    values, categories=self._category_levels[column]
                )
            else:
                matrix[column] = values.fillna("__MISSING__").astype(str)
        for column in set(matrix.columns) - set(self._categorical_columns):
            matrix[column] = pd.to_numeric(matrix[column], errors="coerce")
        return matrix

    def _fit_estimator(self, matrix: pd.DataFrame, target: np.ndarray) -> None:
        """Fit backend-specific estimator with native categorical support."""
        if self._categorical_mode == "lightgbm":
            if self._estimator is None:
                raise RuntimeError("LightGBM estimator was not initialized")
            self._estimator.fit(
                matrix,
                target,
                categorical_feature=self._categorical_columns,
            )
        else:
            categorical_indices = [
                matrix.columns.get_loc(column) for column in self._categorical_columns
            ]
            self._estimator.fit(
                matrix,
                target,
                cat_features=categorical_indices,
                verbose=False,
            )

    def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast all requested series recursively over consecutive dates.

        For each validation day, target lags and rolling features are rebuilt
        from observed training history plus predictions already generated in
        this rollout. Validation actuals are rejected and never appended.
        """
        if self._history is None or self._forecast_origin is None:
            raise RuntimeError("model must be fitted before predict()")
        if not isinstance(future_data, pd.DataFrame):
            raise TypeError("future_data must be a pandas DataFrame")
        if "sales" in future_data.columns or "actual" in future_data.columns:
            raise ValueError("future_data must not contain validation target values")
        required = {*GRAIN_COLUMNS, *self._covariate_columns}
        missing = sorted(required - set(future_data.columns))
        if missing:
            raise ValueError(f"future_data is missing required columns: {missing}")
        future = future_data.loc[:, [*GRAIN_COLUMNS, *self._covariate_columns]].copy()
        future["date"] = pd.to_datetime(future["date"], errors="raise")
        if future.empty or future["date"].isna().any():
            raise ValueError("future_data must contain valid future dates")
        if future.duplicated(GRAIN_COLUMNS).any():
            raise ValueError("future_data contains duplicate date/store/family rows")
        if (future["date"] <= self._forecast_origin).any():
            raise ValueError(
                "future_data dates must be after the fitted forecast origin"
            )

        series_keys = set(
            map(tuple, future.loc[:, SERIES_COLUMNS].drop_duplicates().to_numpy())
        )
        unknown = series_keys - self._series_keys
        if unknown:
            raise ValueError(
                f"future_data contains unseen store-family keys: {unknown}"
            )
        dates = pd.DatetimeIndex(future["date"].drop_duplicates().sort_values())
        expected_dates = pd.date_range(
            self._forecast_origin + pd.Timedelta(days=1), periods=len(dates), freq="D"
        )
        if not dates.equals(expected_dates):
            raise ValueError("future_data must cover consecutive days after the origin")
        expected_set = set(dates)
        for _, group in future.groupby(SERIES_COLUMNS, sort=False):
            if set(group["date"]) != expected_set:
                raise ValueError("each forecast series must contain every horizon date")

        future = future.sort_values(GRAIN_COLUMNS, kind="stable").reset_index(drop=True)
        history_mask = pd.MultiIndex.from_frame(self._history[SERIES_COLUMNS]).isin(
            pd.MultiIndex.from_tuples(series_keys, names=SERIES_COLUMNS)
        )
        context_history = self._history.loc[history_mask].copy()
        recursive_rows: list[pd.DataFrame] = []
        prediction_rows: list[pd.DataFrame] = []
        for date in expected_dates:
            current = future.loc[future["date"].eq(date)].copy()
            current["sales"] = np.nan
            context_parts = [context_history, *recursive_rows, current]
            context = pd.concat(context_parts, ignore_index=True)
            features = build_features(context)
            current_features = features.loc[features["date"].eq(date)].sort_values(
                SERIES_COLUMNS, kind="stable"
            )
            matrix = self._encode_features(current_features)
            values = np.asarray(self._estimator.predict(matrix), dtype=float).reshape(
                -1
            )
            if len(values) != len(current_features):
                raise RuntimeError("estimator returned an unexpected prediction count")
            if not np.isfinite(values).all():
                raise RuntimeError("estimator returned non-finite predictions")
            values = np.maximum(values, 0.0)
            current_features = current_features.loc[:, GRAIN_COLUMNS].copy()
            current_features["prediction"] = values
            prediction_rows.append(current_features)
            predicted_targets = current.loc[
                :, [*GRAIN_COLUMNS, *self._covariate_columns]
            ].copy()
            predicted_targets["sales"] = values
            recursive_rows.append(predicted_targets)

        predictions = pd.concat(prediction_rows, ignore_index=True)
        return (
            predictions.loc[:, PREDICTION_COLUMNS]
            .sort_values(GRAIN_COLUMNS, kind="stable")
            .reset_index(drop=True)
        )
