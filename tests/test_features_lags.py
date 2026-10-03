import pandas as pd

from forecasting.features.lags import add_lag_features


def test_calendar_lags_on_uninterrupted_daily_series() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=15),
            "store_nbr": 1,
            "family": "GROCERY I",
            "sales": range(1, 16),
        }
    )

    result = add_lag_features(data).set_index("date")

    target_date = pd.Timestamp("2020-01-15")
    assert result.loc[target_date, "lag_1"] == 14
    assert result.loc[target_date, "lag_7"] == 8
    assert result.loc[target_date, "lag_14"] == 1


def test_navidad_gap_uses_exact_calendar_date_lags() -> None:
    dates = pd.date_range("2016-12-19", "2016-12-24").append(
        pd.DatetimeIndex(["2016-12-26"])
    )
    data = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 44,
            "family": "GROCERY I",
            "sales": range(1, len(dates) + 1),
        }
    )

    result = add_lag_features(data).set_index("date")
    boxing_day = pd.Timestamp("2016-12-26")

    assert pd.Timestamp("2016-12-25") not in result.index
    assert pd.isna(result.loc[boxing_day, "lag_1"])
    assert result.loc[boxing_day, "lag_7"] == 1


def test_lags_do_not_cross_store_family_boundaries() -> None:
    data = pd.DataFrame(
        {
            "date": list(pd.date_range("2020-01-01", periods=2)) * 2,
            "store_nbr": [1, 1, 2, 2],
            "family": ["GROCERY I"] * 4,
            "sales": [10, 11, 90, 91],
        }
    )

    result = add_lag_features(data)
    second_series_first_row = result.query("store_nbr == 2").iloc[0]

    assert pd.isna(second_series_first_row["lag_1"])
    assert second_series_first_row["lag_1"] != 11


def test_lag_features_sort_rows_and_preserve_input() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-02", "2020-01-01"]),
            "store_nbr": [1, 1],
            "family": ["GROCERY I", "GROCERY I"],
            "sales": [2, 1],
        }
    )
    original = data.copy(deep=True)

    result = add_lag_features(data)

    assert result["date"].is_monotonic_increasing
    assert pd.isna(result.loc[result["sales"].eq(1), "lag_1"]).all()
    pd.testing.assert_frame_equal(data, original)
