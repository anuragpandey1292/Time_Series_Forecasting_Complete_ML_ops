"""Within-series target lag feature generation."""

from __future__ import annotations

import pandas as pd

LAG_PERIODS = (1, 7, 14, 28, 56, 364)
SERIES_COLUMNS = ["store_nbr", "family"]
ORDER_COLUMNS = [*SERIES_COLUMNS, "date"]


def add_lag_features(data: pd.DataFrame, target_column: str = "sales") -> pd.DataFrame:
    """Add observation-based target lags independently for each series.

    Rows are sorted by store, family, and date before shifting. A lag counts
    prior observations in that series; absent calendar dates are not inserted
    or imputed.
    """
    required = {*ORDER_COLUMNS, target_column}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.sort_values(ORDER_COLUMNS, kind="stable").reset_index(drop=True)
    grouped_target = result.groupby(SERIES_COLUMNS, sort=False)[target_column]
    for period in LAG_PERIODS:
        result[f"lag_{period}"] = grouped_target.shift(period)
    return result
