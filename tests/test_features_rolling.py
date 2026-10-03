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


def test_rolling_window_uses_exact_calendar_bounds() -> None:
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-09", "2020-01-10"])
    data = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": [999.0, 1.0, 7.0, 100.0],
        }
    )

    result = add_rolling_features(data)
    row = result.loc[result["date"].eq(pd.Timestamp("2020-01-10"))].iloc[0]

    # Jan 2 is outside [Jan 3, Jan 10); the missing days are not backfilled.
    assert row["rolling_mean_7"] == 4
    assert row["rolling_std_7"] == np.std([1.0, 7.0], ddof=1)


def test_sparse_calendar_window_minimum_observation_rules() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-04", "2020-01-10", "2020-01-11"]),
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": [5.0, 8.0, np.nan],
        }
    )

    result = add_rolling_features(data)
    jan10 = result.loc[result["date"].eq(pd.Timestamp("2020-01-10"))].iloc[0]
    jan11 = result.loc[result["date"].eq(pd.Timestamp("2020-01-11"))].iloc[0]

    assert jan10["rolling_mean_7"] == 5
    assert pd.isna(jan10["rolling_std_7"])
    assert jan11["rolling_mean_7"] == 6.5
    assert jan11["rolling_std_7"] == np.std([5.0, 8.0], ddof=1)


def test_rolling_statistics_do_not_cross_series_boundaries() -> None:
    result = add_rolling_features(_two_series_data())
    first_row = result.query("store_nbr == 2").iloc[0]
    mask = result["date"].eq(pd.Timestamp("2020-01-08"))
    eighth_row = result.loc[result["store_nbr"].eq(2) & mask].iloc[0]

    assert pd.isna(first_row["rolling_mean_7"])
    assert eighth_row["rolling_mean_7"] == 104
    assert eighth_row["rolling_std_7"] == np.std(range(101, 108), ddof=1)


def test_rolling_windows_use_available_observations_inside_calendar_span() -> None:
    result = add_rolling_features(_two_series_data())
    mask = result["store_nbr"].eq(1) & result["date"].eq(pd.Timestamp("2020-01-08"))
    row = result.loc[mask].iloc[0]

    assert row["rolling_mean_14"] == 4
    assert row["rolling_std_14"] == np.std(range(1, 8), ddof=1)
