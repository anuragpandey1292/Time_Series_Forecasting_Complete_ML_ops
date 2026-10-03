"""Promotion feature generation from observed promotion counts."""

from __future__ import annotations

import pandas as pd

SERIES_COLUMNS = ["store_nbr", "family"]
ORDER_COLUMNS = [*SERIES_COLUMNS, "date"]


def add_promotion_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add current, calendar-lagged, and prior calendar-window features.

    The lag uses the promotion count exactly one calendar day earlier.
    Rolling sums cover ``[date - N days, date)`` and require at least one
    non-null observation. Missing dates are not synthesized.
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
    if result["date"].isna().any():
        raise ValueError("data.date must not contain missing values")
    if result.duplicated(subset=[*SERIES_COLUMNS, "date"]).any():
        raise ValueError("data must be unique on date + store_nbr + family")

    result["promotion_today"] = result["onpromotion"]
    result["promotion_lag_1"] = float("nan")
    for window in (7, 14):
        result[f"promotion_rolling_{window}"] = float("nan")
    for _, group in result.groupby(SERIES_COLUMNS, sort=False):
        promotions_by_date = group.set_index("date")["onpromotion"]
        source_dates = group["date"] - pd.Timedelta(days=1)
        result.loc[group.index, "promotion_lag_1"] = source_dates.map(
            promotions_by_date
        )
        promotions = group.set_index("date")["onpromotion"]
        for window in (7, 14):
            result.loc[group.index, f"promotion_rolling_{window}"] = (
                promotions.rolling(f"{window}D", closed="left", min_periods=1)
                .sum()
                .to_numpy()
            )
    return result
