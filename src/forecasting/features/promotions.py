"""Promotion feature generation from observed promotion counts."""

from __future__ import annotations

import pandas as pd

SERIES_COLUMNS = ["store_nbr", "family"]
ORDER_COLUMNS = [*SERIES_COLUMNS, "date"]


def add_promotion_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add current, lagged, and prior-window promotion count features.

    ``promotion_rolling_7`` and ``promotion_rolling_14`` are sums over the
    preceding 7 and 14 observations, respectively; the current promotion is
    excluded from those windows.
    """
    required = {*ORDER_COLUMNS, "onpromotion"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    if not pd.api.types.is_numeric_dtype(data["onpromotion"]):
        raise ValueError("data.onpromotion must be numeric")

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(ORDER_COLUMNS, kind="stable").reset_index(drop=True)
    grouped_promotion = result.groupby(SERIES_COLUMNS, sort=False)["onpromotion"]
    result["promotion_today"] = result["onpromotion"]
    result["promotion_lag_1"] = grouped_promotion.shift(1)
    for window in (7, 14):
        result[f"promotion_rolling_{window}"] = grouped_promotion.transform(
            lambda values, window=window: (
                values.shift(1).rolling(window=window, min_periods=window).sum()
            )
        )
    return result
