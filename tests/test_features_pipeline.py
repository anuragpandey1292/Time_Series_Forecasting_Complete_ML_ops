import numpy as np
import pandas as pd

from forecasting.features.pipeline import build_features


def test_pipeline_preserves_grain_and_does_not_mutate_input() -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=3),
            "store_nbr": [1, 1, 2],
            "family": ["GROCERY I"] * 3,
            "sales": [1.0, np.nan, 3.0],
            "onpromotion": [0, 1, 2],
        }
    )
    original = data.copy(deep=True)

    result = build_features(data)

    assert len(result) == len(data)
    assert not result.duplicated(["date", "store_nbr", "family"]).any()
    assert result["sales"].isna().sum() == 1
    pd.testing.assert_frame_equal(data, original)


def test_pipeline_does_not_fill_or_synthesize_christmas_closure() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-12-24", "2016-12-26"]),
            "store_nbr": [44, 44],
            "family": ["GROCERY I", "GROCERY I"],
            "sales": [12.0, 14.0],
            "onpromotion": [0, 0],
        }
    )

    result = build_features(data)

    assert result["date"].tolist() == list(data["date"])
    assert pd.Timestamp("2016-12-25") not in set(result["date"])
    assert result["sales"].tolist() == [12.0, 14.0]


def test_pipeline_preserves_explicit_missing_christmas_target() -> None:
    data = pd.DataFrame(
        {
            "date": pd.to_datetime(["2016-12-24", "2016-12-25", "2016-12-26"]),
            "store_nbr": [44, 44, 44],
            "family": ["GROCERY I"] * 3,
            "sales": [12.0, np.nan, 14.0],
            "onpromotion": [0, 0, 0],
        }
    )

    result = build_features(data)
    christmas_row = result.loc[result["date"].eq(pd.Timestamp("2016-12-25"))]

    assert len(christmas_row) == 1
    assert pd.isna(christmas_row.iloc[0]["sales"])
    assert christmas_row.iloc[0]["sales"] != 0
