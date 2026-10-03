"""Calendar feature generation from a date column."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_calendar_features(data: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with calendar and cyclical weekday features added.

    Dates are interpreted as calendar dates. ISO week numbers are used for
    ``week``; ``day_of_week`` uses Monday=0 through Sunday=6.
    """
    if "date" not in data.columns:
        raise ValueError("data must contain a 'date' column")

    result = data.copy()
    try:
        result["date"] = pd.to_datetime(result["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("data.date must contain valid dates") from exc
    if result["date"].isna().any():
        raise ValueError("data.date must not contain missing values")

    dates = result["date"].dt
    result["year"] = dates.year
    result["month"] = dates.month
    result["quarter"] = dates.quarter
    result["week"] = dates.isocalendar().week.astype("int16")
    result["day"] = dates.day
    result["day_of_week"] = dates.dayofweek
    result["day_of_year"] = dates.dayofyear
    result["is_weekend"] = dates.dayofweek >= 5
    result["is_month_start"] = dates.is_month_start
    result["is_month_end"] = dates.is_month_end
    result["is_quarter_start"] = dates.is_quarter_start
    result["is_quarter_end"] = dates.is_quarter_end
    result["is_year_start"] = dates.is_year_start
    result["is_year_end"] = dates.is_year_end
    result["dow_sin"] = np.sin(2 * np.pi * result["day_of_week"] / 7)
    result["dow_cos"] = np.cos(2 * np.pi * result["day_of_week"] / 7)
    return result
