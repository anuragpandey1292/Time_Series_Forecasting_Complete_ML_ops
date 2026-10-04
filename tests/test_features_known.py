"""Tests for forecast-time-known exogenous feature preparation."""

import numpy as np
import pandas as pd

from forecasting.features.known import add_known_features, prepare_oil_prices


def test_known_features_align_oil_payday_holiday_and_store_metadata() -> None:
    """Known covariates are aligned by date and scoped to store metadata."""
    dates = pd.to_datetime(["2017-01-15", "2017-01-16"])
    rows = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1, 2],
            "family": ["GROCERY I", "GROCERY I"],
            "onpromotion": [3, 4],
        }
    )
    stores = pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Guayaquil"],
            "state": ["Pichincha", "Guayas"],
            "type": ["A", "B"],
            "cluster": [1, 2],
        }
    )
    holidays = pd.DataFrame(
        {
            "date": [dates[0]],
            "type": ["Holiday"],
            "locale": ["Local"],
            "locale_name": ["Quito"],
            "transferred": [False],
        }
    )
    oil = pd.Series([50.0, 51.0], index=dates)

    result = add_known_features(rows, stores, oil, holidays)

    assert result["oil_price"].tolist() == [50.0, 51.0]
    assert result["holiday_indicator"].tolist() == [1, 0]
    assert result["payday"].tolist() == [1, 0]
    assert result["city"].tolist() == ["Quito", "Guayaquil"]


def test_oil_preparation_preserves_leading_missing_value() -> None:
    """Oil quotes carry forward only after the first known quote."""
    oil = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-03"]),
            "dcoilwtico": [np.nan, 50.0],
        }
    )

    prices = prepare_oil_prices(oil)

    assert pd.isna(prices.loc[pd.Timestamp("2013-01-01")])
    assert pd.isna(prices.loc[pd.Timestamp("2013-01-02")])
    assert prices.loc[pd.Timestamp("2013-01-03")] == 50.0
