"""Leakage-safe within-series calendar-window target statistics."""

from __future__ import annotations

import pandas as pd

ROLLING_WINDOWS = (7, 14, 28, 56)
SERIES_COLUMNS = ["store_nbr", "family"]
ORDER_COLUMNS = [*SERIES_COLUMNS, "date"]


def add_rolling_features(
    data: pd.DataFrame, target_column: str = "sales"
) -> pd.DataFrame:
    """Add statistics over the preceding calendar days, excluding today.

    For a window of ``N`` days at date ``t``, use observed rows in
    ``[t - N days, t)``. Missing dates are not synthesized. Means require at
    least one non-null target in the window; sample standard deviations
    require at least two. Missing targets are never imputed.
    """
    required = {*ORDER_COLUMNS, target_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(ORDER_COLUMNS, kind="stable").reset_index(drop=True)
    if result["date"].isna().any():
        raise ValueError("data.date must not contain missing values")
    if result.duplicated(subset=[*SERIES_COLUMNS, "date"]).any():
        raise ValueError("data must be unique on date + store_nbr + family")

    for window in ROLLING_WINDOWS:
        mean_column = f"rolling_mean_{window}"
        std_column = f"rolling_std_{window}"
        result[mean_column] = float("nan")
        result[std_column] = float("nan")
        for _, group in result.groupby(SERIES_COLUMNS, sort=False):
            values = group.set_index("date")[target_column]
            rolling = values.rolling(f"{window}D", closed="left", min_periods=1)
            result.loc[group.index, mean_column] = rolling.mean().to_numpy()
            result.loc[group.index, std_column] = (
                values.rolling(f"{window}D", closed="left", min_periods=2)
                .std()
                .to_numpy()
            )
    return result
