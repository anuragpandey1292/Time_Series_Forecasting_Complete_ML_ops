"""Tests for explicit training-only target and exogenous imputation."""

import numpy as np
import pandas as pd
import pytest

from forecasting.data.imputation import (
    impute_training_exogenous,
    impute_training_sales,
    impute_training_sales_for_modeling,
)

STORES = pd.DataFrame(
    {
        "store_nbr": [1, 2, 3],
        "city": ["Quito", "Guayaquil", "Cuenca"],
        "state": ["Pichincha", "Guayas", "Azuay"],
    }
)
HOLIDAY_COLUMNS = [
    "date",
    "type",
    "locale",
    "locale_name",
    "description",
    "transferred",
]


def make_holidays(rows: list[tuple]) -> pd.DataFrame:
    """Make a holiday frame with the Favorita holiday schema."""
    return pd.DataFrame(rows, columns=HOLIDAY_COLUMNS)


def make_sales(rows: list[tuple]) -> pd.DataFrame:
    """Make a small target frame from (date, store, family, sales) tuples."""
    return pd.DataFrame(rows, columns=["date", "store_nbr", "family", "sales"])


def impute(
    data: pd.DataFrame,
    holidays: pd.DataFrame | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Call the training imputer with shared small metadata fixtures."""
    default_end = pd.to_datetime(data["date"], errors="raise").max()
    return impute_training_sales(
        data,
        STORES,
        holidays if holidays is not None else make_holidays([]),
        train_end=kwargs.pop("train_end", default_end),
        **kwargs,
    )


def test_complete_daily_grid_and_sales_audit_fields() -> None:
    """Missing calendar rows are created and source values remain auditable."""
    data = make_sales(
        [("2020-01-01", 1, "GROCERY I", 10.0), ("2020-01-03", 1, "GROCERY I", 30.0)]
    )
    original = data.copy(deep=True)

    result = impute(data)

    assert result["date"].tolist() == list(pd.date_range("2020-01-01", periods=3))
    middle = result.loc[result["date"].eq(pd.Timestamp("2020-01-02"))].iloc[0]
    assert middle["sales"] == 10
    assert middle["sales_original"] != middle["sales_original"]
    assert middle["sales_imputation_method"] == "ffill"
    assert middle["is_imputed"]
    assert (
        result.loc[
            result["date"].eq(pd.Timestamp("2020-01-01")), "sales_original"
        ].iloc[0]
        == 10
    )
    assert (
        result.loc[
            result["date"].eq(pd.Timestamp("2020-01-01")), "sales_imputation_method"
        ].iloc[0]
        == "observed"
    )
    assert not result.loc[
        result["date"].eq(pd.Timestamp("2020-01-01")), "is_imputed"
    ].iloc[0]
    pd.testing.assert_frame_equal(data, original)


def test_national_holiday_applies_to_every_store() -> None:
    """A confirmed national closure imputes zero for each applicable store."""
    data = make_sales(
        [
            ("2020-01-01", 1, "GROCERY I", 10.0),
            ("2020-01-03", 1, "GROCERY I", 12.0),
            ("2020-01-01", 2, "GROCERY I", 20.0),
            ("2020-01-03", 2, "GROCERY I", 22.0),
        ]
    )
    holiday = make_holidays(
        [("2020-01-02", "Holiday", "National", "Ecuador", "National day", False)]
    )

    result = impute(data, holiday)
    missing_day = result.loc[result["date"].eq(pd.Timestamp("2020-01-02"))]

    assert missing_day["sales"].tolist() == [0, 0]
    assert missing_day["sales_imputation_method"].tolist() == ["holiday_zero"] * 2


def test_regional_holiday_applies_only_to_matching_state() -> None:
    """Regional closure scope matches stores.state only."""
    data = make_sales(
        [
            ("2020-01-01", 1, "GROCERY I", 10.0),
            ("2020-01-03", 1, "GROCERY I", 12.0),
            ("2020-01-01", 2, "GROCERY I", 20.0),
            ("2020-01-03", 2, "GROCERY I", 22.0),
        ]
    )
    holiday = make_holidays(
        [("2020-01-02", "Holiday", "Regional", "Pichincha", "Regional day", False)]
    )

    result = impute(data, holiday)
    missing_day = result.loc[result["date"].eq(pd.Timestamp("2020-01-02"))].set_index(
        "store_nbr"
    )

    assert missing_day.loc[1, "sales"] == 0
    assert missing_day.loc[2, "sales"] == 20
    assert missing_day.loc[2, "sales_imputation_method"] == "ffill"


def test_local_holiday_applies_only_to_matching_city() -> None:
    """Local closure scope matches stores.city only."""
    data = make_sales(
        [
            ("2020-01-01", 1, "GROCERY I", 10.0),
            ("2020-01-03", 1, "GROCERY I", 12.0),
            ("2020-01-01", 2, "GROCERY I", 20.0),
            ("2020-01-03", 2, "GROCERY I", 22.0),
        ]
    )
    holiday = make_holidays(
        [("2020-01-02", "Holiday", "Local", "Quito", "City day", False)]
    )

    result = impute(data, holiday)
    missing_day = result.loc[result["date"].eq(pd.Timestamp("2020-01-02"))].set_index(
        "store_nbr"
    )

    assert missing_day.loc[1, "sales"] == 0
    assert missing_day.loc[2, "sales"] == 20


def test_national_navidad_missing_sales_become_holiday_zero() -> None:
    """The raw-schema Navidad holiday on Dec 25 is recognized nationally."""
    data = make_sales(
        [
            ("2016-12-24", 1, "GROCERY I", 100.0),
            ("2016-12-26", 1, "GROCERY I", 120.0),
        ]
    )
    holiday = make_holidays(
        [("2016-12-25", "Holiday", "National", "Ecuador", "Navidad", False)]
    )

    result = impute(data, holiday, train_end="2016-12-26")
    christmas = result.loc[result["date"].eq(pd.Timestamp("2016-12-25"))].iloc[0]

    assert christmas["sales"] == 0
    assert christmas["sales_original"] != christmas["sales_original"]
    assert christmas["sales_imputation_method"] == "holiday_zero"
    assert christmas["is_imputed"]


def test_leading_training_gaps_are_backfilled_after_forward_fill() -> None:
    """Leading gaps use the first training observation when no prior value exists."""
    data = make_sales(
        [("2020-01-02", 1, "GROCERY I", 20.0), ("2020-01-03", 1, "GROCERY I", 30.0)]
    )

    result = impute(data, train_start="2020-01-01")
    leading = result.loc[result["date"].eq(pd.Timestamp("2020-01-01"))].iloc[0]

    assert leading["sales"] == 20
    assert leading["sales_imputation_method"] == "bfill"
    assert leading["is_imputed"]


def test_imputation_never_crosses_store_or_family_boundaries() -> None:
    """Each series fills from its own values, never a neighboring series."""
    data = make_sales(
        [
            ("2020-01-01", 1, "GROCERY I", 10.0),
            ("2020-01-03", 1, "GROCERY I", 30.0),
            ("2020-01-01", 2, "GROCERY I", 200.0),
            ("2020-01-03", 2, "GROCERY I", 300.0),
            ("2020-01-01", 1, "PRODUCE", 1000.0),
            ("2020-01-03", 1, "PRODUCE", 3000.0),
        ]
    )

    result = impute(data)
    middle = result.loc[result["date"].eq(pd.Timestamp("2020-01-02"))].set_index(
        ["store_nbr", "family"]
    )

    assert middle.loc[(1, "GROCERY I"), "sales"] == 10
    assert middle.loc[(2, "GROCERY I"), "sales"] == 200
    assert middle.loc[(1, "PRODUCE"), "sales"] == 1000


def test_transferred_source_date_is_not_closure_but_transfer_date_is() -> None:
    """The original holiday is skipped; its actual Transfer date is closed."""
    data = make_sales(
        [("2020-01-01", 1, "GROCERY I", 5.0), ("2020-01-04", 1, "GROCERY I", 9.0)]
    )
    holidays = make_holidays(
        [
            ("2020-01-02", "Holiday", "National", "Ecuador", "Moved day", True),
            ("2020-01-03", "Transfer", "National", "Ecuador", "Moved day", False),
        ]
    )

    result = impute(data, holidays)
    jan2 = result.loc[result["date"].eq(pd.Timestamp("2020-01-02"))].iloc[0]
    jan3 = result.loc[result["date"].eq(pd.Timestamp("2020-01-03"))].iloc[0]

    assert jan2["sales"] == 5
    assert jan2["sales_imputation_method"] == "ffill"
    assert jan3["sales"] == 0
    assert jan3["sales_imputation_method"] == "holiday_zero"


def test_bridge_is_closure_but_work_day_and_generic_event_are_not() -> None:
    """Event meanings are explicit rather than treating all records as closure."""
    data = make_sales(
        [
            ("2020-01-01", 1, "GROCERY I", 10.0),
            ("2020-01-03", 1, "GROCERY I", 12.0),
            ("2020-01-05", 1, "GROCERY I", 14.0),
            ("2020-01-07", 1, "GROCERY I", 16.0),
        ]
    )
    holidays = make_holidays(
        [
            ("2020-01-02", "Work Day", "National", "Ecuador", "Make-up", False),
            ("2020-01-04", "Bridge", "National", "Ecuador", "Bridge", False),
            ("2020-01-06", "Event", "National", "Ecuador", "Event", False),
        ]
    )

    result = impute(data, holidays).set_index("date")

    assert result.loc[pd.Timestamp("2020-01-02"), "sales"] == 10
    assert result.loc[pd.Timestamp("2020-01-02"), "sales_imputation_method"] == "ffill"
    assert result.loc[pd.Timestamp("2020-01-04"), "sales"] == 0
    assert (
        result.loc[pd.Timestamp("2020-01-04"), "sales_imputation_method"]
        == "holiday_zero"
    )
    assert result.loc[pd.Timestamp("2020-01-06"), "sales"] == 14
    assert result.loc[pd.Timestamp("2020-01-06"), "sales_imputation_method"] == "ffill"


def test_validation_or_test_targets_after_train_end_are_rejected() -> None:
    """The explicit training cutoff prevents imputation over later actuals."""
    data = make_sales(
        [("2020-01-01", 1, "GROCERY I", 10.0), ("2020-01-04", 1, "GROCERY I", np.nan)]
    )

    with pytest.raises(ValueError, match="after train_end"):
        impute(data, train_end="2020-01-03")


def test_modeling_policy_keeps_holiday_zero_and_forward_fills_ordinary_gap() -> None:
    """Modeling policy retains closures and uses only preceding sales."""
    data = make_sales(
        [
            ("2020-01-01", 1, "GROCERY I", 10.0),
            ("2020-01-04", 1, "GROCERY I", 40.0),
            ("2020-01-06", 1, "GROCERY I", 60.0),
        ]
    )
    holidays = make_holidays(
        [("2020-01-03", "Holiday", "National", "Ecuador", "Closure", False)]
    )

    result = impute_training_sales_for_modeling(
        data,
        STORES,
        holidays,
        train_end="2020-01-06",
    ).set_index("date")

    assert result.loc[pd.Timestamp("2020-01-02"), "sales"] == 10
    assert result.loc[pd.Timestamp("2020-01-03"), "sales"] == 0
    assert result.loc[pd.Timestamp("2020-01-04"), "sales"] == 40
    assert result.loc[pd.Timestamp("2020-01-05"), "sales"] == 40
    assert (
        result.loc[pd.Timestamp("2020-01-03"), "sales_imputation_method"]
        == "holiday_zero"
    )
    assert (
        result.loc[pd.Timestamp("2020-01-04"), "sales_imputation_method"] == "observed"
    )
    assert result.loc[pd.Timestamp("2020-01-05"), "sales_imputation_method"] == "ffill"
    assert pd.isna(result.loc[pd.Timestamp("2020-01-03"), "sales_original"])
    assert result.loc[pd.Timestamp("2020-01-04"), "sales_original"] == 40
    assert result.loc[pd.Timestamp("2020-01-03"), "is_imputed"]
    assert not result.loc[pd.Timestamp("2020-01-04"), "is_imputed"]


def test_modeling_policy_leaves_leading_gap_unfilled_despite_later_sales() -> None:
    """A later training observation is never used to fill an earlier target."""
    data = make_sales(
        [("2020-01-02", 1, "GROCERY I", 20.0), ("2020-01-03", 1, "GROCERY I", 30.0)]
    )

    result = impute_training_sales_for_modeling(
        data,
        STORES,
        make_holidays([]),
        train_start="2020-01-01",
        train_end="2020-01-03",
    )
    leading = result.loc[result["date"].eq(pd.Timestamp("2020-01-01"))].iloc[0]

    assert pd.isna(leading["sales"])
    assert pd.isna(leading["sales_original"])
    assert leading["sales_imputation_method"] == "unfilled"
    assert not leading["is_imputed"]


def test_modeling_imputation_is_repeated_at_each_fold_boundary() -> None:
    """Fold-local calls cannot see later targets or cross their train_end."""
    full_history = make_sales(
        [
            ("2020-01-01", 1, "GROCERY I", np.nan),
            ("2020-01-02", 1, "GROCERY I", 20.0),
            ("2020-01-03", 1, "GROCERY I", 30.0),
        ]
    )
    holidays = make_holidays([])
    fold_outputs = []
    for origin in ("2020-01-01", "2020-01-02"):
        fold_end = pd.Timestamp(origin)
        fold_train = full_history.loc[pd.to_datetime(full_history["date"]).le(fold_end)]
        fold_outputs.append(
            impute_training_sales_for_modeling(
                fold_train,
                STORES,
                holidays,
                train_start="2020-01-01",
                train_end=fold_end,
            )
        )

    assert all(pd.isna(output.loc[0, "sales"]) for output in fold_outputs)
    with pytest.raises(ValueError, match="after train_end"):
        impute_training_sales_for_modeling(
            full_history,
            STORES,
            holidays,
            train_start="2020-01-01",
            train_end="2020-01-01",
        )


def test_exogenous_imputation_is_separate_and_audited() -> None:
    """Exogenous ffill/bfill has per-column audit flags and no target logic."""
    data = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=4),
            "store_nbr": 1,
            "oil_price": [np.nan, 10.0, np.nan, 13.0],
        }
    )
    original = data.copy(deep=True)

    result = impute_training_exogenous(
        data, ["oil_price"], train_end="2020-01-04", group_columns=["store_nbr"]
    )

    assert result["oil_price"].tolist() == [10.0, 10.0, 10.0, 13.0]
    assert result["oil_price_was_imputed"].tolist() == [True, False, True, False]
    pd.testing.assert_frame_equal(data, original)
    with pytest.raises(ValueError, match="use impute_training_sales"):
        impute_training_exogenous(
            data.assign(sales=1), ["sales"], train_end="2020-01-04"
        )
