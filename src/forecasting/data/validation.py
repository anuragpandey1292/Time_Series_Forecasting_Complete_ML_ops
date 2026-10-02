"""Lightweight validation checks for the raw Favorita datasets."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from forecasting.data.loader import DATASET_FILES, load_dataset

TRAIN_COLUMNS = {
    "id",
    "date",
    "store_nbr",
    "family",
    "sales",
    "onpromotion",
}
TEST_COLUMNS = {"id", "date", "store_nbr", "family", "onpromotion"}
KEY_COLUMNS = ["date", "store_nbr", "family"]


def require_columns(
    frame: pd.DataFrame, required: Iterable[str], dataset_name: str
) -> None:
    """Raise when a dataset does not contain all required columns."""
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def validate_train_schema(train: pd.DataFrame) -> None:
    """Validate that train has exactly the expected columns."""
    require_columns(train, TRAIN_COLUMNS, "train")
    unexpected = sorted(set(train.columns) - TRAIN_COLUMNS)
    if unexpected:
        raise ValueError(f"train has unexpected columns: {unexpected}")


def validate_test_schema(test: pd.DataFrame) -> None:
    """Validate that test has exactly the expected columns."""
    require_columns(test, TEST_COLUMNS, "test")
    unexpected = sorted(set(test.columns) - TEST_COLUMNS)
    if unexpected:
        raise ValueError(f"test has unexpected columns: {unexpected}")


def validate_non_null(
    frame: pd.DataFrame, columns: Iterable[str], dataset_name: str
) -> None:
    """Raise if any specified columns contain missing values."""
    column_names = list(columns)
    require_columns(frame, column_names, dataset_name)
    missing_counts = frame[column_names].isna().sum()
    invalid = missing_counts[missing_counts > 0]
    if not invalid.empty:
        details = ", ".join(f"{column}={count}" for column, count in invalid.items())
        raise ValueError(f"{dataset_name} contains missing values: {details}")


def validate_non_negative(
    frame: pd.DataFrame, column: str, dataset_name: str
) -> None:
    """Raise if a numeric column contains values below zero."""
    require_columns(frame, [column], dataset_name)
    negative_count = int((frame[column] < 0).sum())
    if negative_count:
        raise ValueError(
            f"{dataset_name}.{column} contains {negative_count} negative value(s)"
        )


def validate_no_duplicate_keys(
    frame: pd.DataFrame, columns: Iterable[str], dataset_name: str
) -> None:
    """Raise if combinations of the specified key columns are repeated."""
    key_columns = list(columns)
    require_columns(frame, key_columns, dataset_name)
    duplicate_count = int(frame.duplicated(subset=key_columns).sum())
    if duplicate_count:
        raise ValueError(
            f"{dataset_name} contains {duplicate_count} duplicate "
            f"{'+'.join(key_columns)} combination(s)"
        )


def validate_store_references(
    train: pd.DataFrame, test: pd.DataFrame, stores: pd.DataFrame
) -> None:
    """Ensure train and test store numbers are present in stores metadata."""
    require_columns(train, ["store_nbr"], "train")
    require_columns(test, ["store_nbr"], "test")
    require_columns(stores, ["store_nbr"], "stores")
    known_stores = set(stores["store_nbr"].dropna())
    invalid_train = set(train["store_nbr"].dropna()) - known_stores
    invalid_test = set(test["store_nbr"].dropna()) - known_stores
    if invalid_train or invalid_test:
        details = []
        if invalid_train:
            details.append(f"train={sorted(invalid_train)}")
        if invalid_test:
            details.append(f"test={sorted(invalid_test)}")
        raise ValueError("Unknown store_nbr values: " + "; ".join(details))


def validate_temporal_order(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Require the latest training date to precede the earliest test date."""
    require_columns(train, ["date"], "train")
    require_columns(test, ["date"], "test")
    train_dates = pd.to_datetime(train["date"], errors="raise")
    test_dates = pd.to_datetime(test["date"], errors="raise")
    if train_dates.isna().any() or test_dates.isna().any():
        raise ValueError("train.date and test.date must not contain missing values")
    if train_dates.max() >= test_dates.min():
        raise ValueError(
            "Maximum train date must be earlier than minimum test date "
            f"(train={train_dates.max()}, test={test_dates.min()})"
        )


def validate_supporting_datasets(
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays_events: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    """Validate basic required fields and store-key uniqueness in support data."""
    require_columns(stores, ["store_nbr"], "stores")
    if stores["store_nbr"].duplicated().any():
        raise ValueError("stores.store_nbr values must be unique")
    require_columns(oil, ["date"], "oil")
    require_columns(holidays_events, ["date"], "holidays_events")
    require_columns(transactions, ["date", "store_nbr"], "transactions")


def validate_datasets(
    train: pd.DataFrame,
    test: pd.DataFrame,
    stores: pd.DataFrame,
    oil: pd.DataFrame,
    holidays_events: pd.DataFrame,
    transactions: pd.DataFrame,
) -> None:
    """Run all M2 validations against the supplied raw dataset frames."""
    validate_train_schema(train)
    validate_test_schema(test)
    validate_non_null(train, KEY_COLUMNS + ["sales", "onpromotion"], "train")
    validate_non_null(test, KEY_COLUMNS, "test")
    validate_non_negative(train, "sales", "train")
    validate_non_negative(train, "onpromotion", "train")
    validate_non_negative(test, "onpromotion", "test")
    validate_no_duplicate_keys(train, KEY_COLUMNS, "train")
    validate_store_references(train, test, stores)
    validate_temporal_order(train, test)
    validate_supporting_datasets(stores, oil, holidays_events, transactions)


def validate_raw_datasets() -> None:
    """Load the six raw CSVs with the existing loader and validate them."""
    datasets = {name: load_dataset(name) for name in DATASET_FILES}
    validate_datasets(
        train=datasets["train"],
        test=datasets["test"],
        stores=datasets["stores"],
        oil=datasets["oil"],
        holidays_events=datasets["holidays_events"],
        transactions=datasets["transactions"],
    )
