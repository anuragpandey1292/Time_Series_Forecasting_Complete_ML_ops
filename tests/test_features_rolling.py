import numpy as np
import pandas as pd

from forecasting.features.rolling import add_rolling_features


def _two_series_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": list(pd.date_range("2020-01-01", periods=9)) * 2,
            "store_nbr": [1] * 9 + [2] * 9,
            "family": ["GROCERY I"] * 18,
            "sales": list(range(1, 10)) + list(range(101, 110)),
        }
    )


def test_rolling_mean_excludes_current_target() -> None:
    result = add_rolling_features(_two_series_data())
    mask = result["store_nbr"].eq(1) & result["date"].eq(pd.Timestamp("2020-01-08"))
    row = result.loc[mask].iloc[0]

    assert row["sales"] == 8
    assert row["rolling_mean_7"] == 4
    assert row["rolling_mean_7"] != np.mean([2, 3, 4, 5, 6, 7, 8])


def test_rolling_statistics_do_not_cross_series_boundaries() -> None:
    result = add_rolling_features(_two_series_data())
    first_row = result.query("store_nbr == 2").iloc[0]
    mask = result["date"].eq(pd.Timestamp("2020-01-08"))
    eighth_row = result.loc[result["store_nbr"].eq(2) & mask].iloc[0]

    assert pd.isna(first_row["rolling_mean_7"])
    assert eighth_row["rolling_mean_7"] == 104
    assert eighth_row["rolling_std_7"] == np.std(range(101, 108), ddof=1)


def test_rolling_windows_require_full_prior_window() -> None:
    result = add_rolling_features(_two_series_data())
    mask = result["store_nbr"].eq(1) & result["date"].eq(pd.Timestamp("2020-01-08"))
    row = result.loc[mask].iloc[0]

    assert pd.isna(row["rolling_mean_14"])
    assert pd.isna(row["rolling_std_14"])
