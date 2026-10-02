"""Unit tests for lightweight Favorita data validation."""

import pandas as pd
import pytest

from forecasting.data.validation import (
    validate_datasets,
    validate_no_duplicate_keys,
    validate_non_negative,
    validate_non_null,
    validate_store_references,
    validate_supporting_datasets,
    validate_temporal_order,
    validate_test_schema,
    validate_train_schema,
)


def valid_datasets() -> dict[str, pd.DataFrame]:
    """Build a small valid set of Favorita-like dataframes in memory."""
    return {
        "train": pd.DataFrame(
            {
                "id": [1, 2],
                "date": ["2017-01-01", "2017-01-02"],
                "store_nbr": [1, 2],
                "family": ["FOODS", "BEVERAGES"],
                "sales": [10.0, 20.0],
                "onpromotion": [0, 3],
            }
        ),
        "test": pd.DataFrame(
            {
                "id": [3],
                "date": ["2017-01-03"],
                "store_nbr": [1],
                "family": ["FOODS"],
                "onpromotion": [1],
            }
        ),
        "stores": pd.DataFrame({"store_nbr": [1, 2]}),
        "oil": pd.DataFrame({"date": ["2017-01-01"], "dcoilwtico": [50.0]}),
        "holidays_events": pd.DataFrame(
            {"date": ["2017-01-01"], "type": ["Holiday"]}
        ),
        "transactions": pd.DataFrame(
            {"date": ["2017-01-01"], "store_nbr": [1], "transactions": [100]}
        ),
    }


def test_valid_datasets_pass_all_checks() -> None:
    """A valid in-memory dataset collection passes the combined validator."""
    validate_datasets(**valid_datasets())


def test_train_schema_rejects_missing_column() -> None:
    """The train schema error names a required column that is absent."""
    train = valid_datasets()["train"].drop(columns="sales")

    with pytest.raises(ValueError, match="sales"):
        validate_train_schema(train)


def test_test_schema_rejects_unexpected_column() -> None:
    """The test schema rejects columns outside the expected schema."""
    test = valid_datasets()["test"].assign(extra=1)

    with pytest.raises(ValueError, match="unexpected columns"):
        validate_test_schema(test)


@pytest.mark.parametrize(
    ("column", "value", "dataset_name"),
    [("sales", -0.1, "train"), ("onpromotion", -1, "test")],
)
def test_negative_domain_values_fail(
    column: str, value: float, dataset_name: str
) -> None:
    """Negative sales or promotion values are rejected."""
    datasets = valid_datasets()
    frame = datasets[dataset_name]
    frame.loc[0, column] = value

    with pytest.raises(ValueError, match="negative"):
        validate_non_negative(frame, column, dataset_name)


def test_combined_validator_rejects_negative_sales() -> None:
    """The combined validator rejects negative train sales."""
    datasets = valid_datasets()
    datasets["train"].loc[0, "sales"] = -0.1

    with pytest.raises(ValueError, match="train.sales.*negative"):
        validate_datasets(**datasets)


def test_train_onpromotion_negative_value_fails() -> None:
    """Negative promotion values are rejected in train as well as test."""
    train = valid_datasets()["train"]
    train.loc[0, "onpromotion"] = -1

    with pytest.raises(ValueError, match="onpromotion"):
        validate_datasets(
            train=train,
            **{key: value for key, value in valid_datasets().items() if key != "train"},
        )


def test_key_column_null_fails() -> None:
    """Missing key values fail without being filled or changed."""
    train = valid_datasets()["train"]
    train.loc[0, "family"] = None

    with pytest.raises(ValueError, match="family"):
        validate_non_null(train, ["date", "store_nbr", "family"], "train")


def test_duplicate_train_key_fails() -> None:
    """Repeated date/store/family keys are rejected."""
    train = valid_datasets()["train"]
    train.loc[1, ["date", "store_nbr", "family"]] = train.loc[
        0, ["date", "store_nbr", "family"]
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_no_duplicate_keys(
            train, ["date", "store_nbr", "family"], "train"
        )


def test_combined_validator_rejects_duplicate_train_key() -> None:
    """The combined validator rejects duplicate train date/store/family keys."""
    datasets = valid_datasets()
    datasets["train"].loc[1, ["date", "store_nbr", "family"]] = datasets[
        "train"
    ].loc[0, ["date", "store_nbr", "family"]]

    with pytest.raises(ValueError, match="duplicate"):
        validate_datasets(**datasets)


def test_unknown_store_reference_fails() -> None:
    """Train or test cannot reference a store absent from stores.csv."""
    datasets = valid_datasets()
    datasets["test"].loc[0, "store_nbr"] = 99

    with pytest.raises(ValueError, match="Unknown store_nbr"):
        validate_store_references(
            datasets["train"], datasets["test"], datasets["stores"]
        )


@pytest.mark.parametrize("test_date", ["2017-01-02", "2016-12-31"])
def test_train_date_must_precede_test_date(test_date: str) -> None:
    """Equal or earlier test dates fail the strict temporal boundary."""
    datasets = valid_datasets()
    datasets["test"].loc[0, "date"] = test_date

    with pytest.raises(ValueError, match="Maximum train date"):
        validate_temporal_order(datasets["train"], datasets["test"])


def test_supporting_dataset_requirements_fail() -> None:
    """Supporting table key/date requirements and store uniqueness are checked."""
    datasets = valid_datasets()
    datasets["stores"] = pd.DataFrame({"store_nbr": [1, 1]})

    with pytest.raises(ValueError, match="unique"):
        validate_supporting_datasets(
            datasets["stores"],
            datasets["oil"],
            datasets["holidays_events"],
            datasets["transactions"],
        )
