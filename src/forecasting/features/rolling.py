"""Leakage-safe within-series rolling target statistics."""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOWS = (7, 14, 28, 56)
SERIES_COLUMNS = ["store_nbr", "family"]
ORDER_COLUMNS = [*SERIES_COLUMNS, "date"]


def add_rolling_features(
    data: pd.DataFrame, target_column: str = "sales"
) -> pd.DataFrame:
    """Add prior-observation rolling means and sample standard deviations.

    The target is shifted by one observation within each store-family series
    before each full-window statistic is calculated, excluding the current
    target and preserving missing target values.
    """
    required = {*ORDER_COLUMNS, target_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(ORDER_COLUMNS, kind="stable").reset_index(drop=True)
    for window in ROLLING_WINDOWS:
        result[f"rolling_mean_{window}"] = result.groupby(SERIES_COLUMNS, sort=False)[
            target_column
        ].transform(
            lambda values, window=window: (
                values.shift(1).rolling(window=window, min_periods=window).mean()
            )
        )
        result[f"rolling_std_{window}"] = result.groupby(SERIES_COLUMNS, sort=False)[
            target_column
        ].transform(
            lambda values, window=window: (
                values.shift(1).rolling(window=window, min_periods=window).std()
            )
        )
    return result
