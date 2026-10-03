"""Composition of the initial calendar, target, and promotion features."""

from __future__ import annotations

import pandas as pd

from forecasting.features.calendar import add_calendar_features
from forecasting.features.lags import add_lag_features
from forecasting.features.promotions import add_promotion_features
from forecasting.features.rolling import add_rolling_features

GRAIN_COLUMNS = ["date", "store_nbr", "family"]


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build the supported feature set without changing or expanding input rows.

    The input must contain one row per date/store/family. No missing sales are
    filled, and dates absent from the input (including closure days) are not
    synthesized.
    """
    missing = sorted(set(GRAIN_COLUMNS) - set(data.columns))
    if missing:
        raise ValueError(f"data is missing grain columns: {missing}")
    if data.duplicated(subset=GRAIN_COLUMNS).any():
        raise ValueError("data must be unique on date + store_nbr + family")

    features = add_calendar_features(data)
    features = add_lag_features(features)
    features = add_rolling_features(features)
    features = add_promotion_features(features)
    features = features.sort_values(
        ["store_nbr", "family", "date"], kind="stable"
    ).reset_index(drop=True)
    if features.duplicated(subset=GRAIN_COLUMNS).any():
        raise ValueError("feature output is not unique on date + store_nbr + family")
    return features
