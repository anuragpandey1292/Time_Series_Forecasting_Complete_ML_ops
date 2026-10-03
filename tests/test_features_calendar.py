import numpy as np
import pandas as pd

from forecasting.features.calendar import add_calendar_features


def test_calendar_features_for_known_date() -> None:
    data = pd.DataFrame({"date": ["2020-01-01"]})

    result = add_calendar_features(data)
    row = result.iloc[0]

    assert row["year"] == 2020
    assert row["month"] == 1
    assert row["quarter"] == 1
    assert row["week"] == 1
    assert row["day"] == 1
    assert row["day_of_week"] == 2
    assert row["day_of_year"] == 1
    assert not row["is_weekend"]
    assert row["is_month_start"]
    assert not row["is_month_end"]
    assert row["is_quarter_start"]
    assert not row["is_quarter_end"]
    assert row["is_year_start"]
    assert not row["is_year_end"]


def test_day_of_week_cyclical_features() -> None:
    result = add_calendar_features(pd.DataFrame({"date": ["2020-01-06"]}))

    assert result.loc[0, "day_of_week"] == 0
    assert result.loc[0, "dow_sin"] == 0.0
    assert result.loc[0, "dow_cos"] == 1.0


def test_calendar_features_do_not_mutate_input() -> None:
    data = pd.DataFrame({"date": ["2020-01-01"]})
    original = data.copy(deep=True)

    add_calendar_features(data)

    pd.testing.assert_frame_equal(data, original)


def test_weekday_cycle_values_are_on_unit_circle() -> None:
    dates = pd.date_range("2020-01-06", periods=7)
    result = add_calendar_features(pd.DataFrame({"date": dates}))

    np.testing.assert_allclose(result["dow_sin"] ** 2 + result["dow_cos"] ** 2, 1)
