import pandas as pd

from forecasting.features.promotions import add_promotion_features


def test_promotion_features_use_calendar_dates_and_prior_windows() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=15),
            "store_nbr": 1,
            "family": "GROCERY I",
            "onpromotion": range(1, 16),
        }
    )

    result = add_promotion_features(data)
    row = result.loc[result["date"].eq(pd.Timestamp("2020-01-15"))].iloc[0]

    assert row["promotion_today"] == 15
    assert row["promotion_lag_1"] == 14
    assert row["promotion_rolling_7"] == sum(range(8, 15))
    assert row["promotion_rolling_14"] == sum(range(1, 15))


def test_promotion_features_respect_missing_calendar_date() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-03", "2020-01-10"])
    data = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": 1,
            "family": "GROCERY I",
            "onpromotion": [100, 3, 10],
        }
    )

    result = add_promotion_features(data)
    jan10 = result.loc[result["date"].eq(pd.Timestamp("2020-01-10"))].iloc[0]

    assert jan10["promotion_rolling_7"] == 3


def test_promotion_features_are_series_specific_and_preserve_input() -> None:
    dates = list(pd.date_range("2020-01-01", periods=8))
    data = pd.DataFrame(
        {
            "date": dates + dates,
            "store_nbr": [1] * 8 + [2] * 8,
            "family": ["GROCERY I"] * 16,
            "onpromotion": [1] * 8 + [9] * 8,
        }
    )
    original = data.copy(deep=True)

    result = add_promotion_features(data)
    second_series_first = result.query("store_nbr == 2").iloc[0]
    mask = result["date"].eq(pd.Timestamp("2020-01-08"))
    second_series_eighth = result.loc[result["store_nbr"].eq(2) & mask].iloc[0]

    assert pd.isna(second_series_first["promotion_lag_1"])
    assert second_series_eighth["promotion_rolling_7"] == 63
    pd.testing.assert_frame_equal(data, original)
