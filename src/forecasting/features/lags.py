"""Calendar-date target lag feature generation."""

from __future__ import annotations

import pandas as pd

LAG_PERIODS = (1, 7, 14, 28, 56, 364)
SERIES_COLUMNS = ["store_nbr", "family"]
ORDER_COLUMNS = [*SERIES_COLUMNS, "date"]


def add_lag_features(data: pd.DataFrame, target_column: str = "sales") -> pd.DataFrame:
    """Add exact calendar-date target lags independently for each series.

    A lag looks up the target exactly ``period`` calendar days before each
    row's date. If that date is absent, the lag is NaN. No rows or values are
    inserted or imputed.
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

    for _, group in result.groupby(SERIES_COLUMNS, sort=False):
        sales_by_date = group.set_index("date")[target_column]
        for period in LAG_PERIODS:
            source_dates = group["date"] - pd.to_timedelta(period, unit="D")
            result.loc[group.index, f"lag_{period}"] = source_dates.map(sales_by_date)
    return result
