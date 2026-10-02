"""Print a concise profile of the raw Favorita CSV datasets."""

from __future__ import annotations

import pandas as pd

from forecasting.data.loader import DATASET_FILES, RAW_DATA_DIR, load_dataset

CHUNK_SIZE = 100_000
TRAIN_TEST_KEY_COLUMNS = ["date", "store_nbr", "family"]


def print_dataset_profile(name: str, frame: pd.DataFrame) -> None:
    """Print dimensions, columns, dtypes, and missing values for a dataset."""
    print(f"\n{'=' * 16} {name} {'=' * 16}")
    print(f"Rows: {len(frame):,}")
    print(f"Columns: {len(frame.columns):,}")
    print("Column details:")
    for column in frame.columns:
        print(
            f"  {column}: dtype={frame[column].dtype}, "
            f"missing={frame[column].isna().sum():,}"
        )


def print_date_and_cardinality_profile(
    frame: pd.DataFrame,
) -> set[pd.Timestamp]:
    """Print date and store/family cardinalities and return the dates."""
    dates = pd.to_datetime(frame["date"], errors="coerce")
    print(f"Minimum date: {dates.min()}")
    print(f"Maximum date: {dates.max()}")
    print(f"Unique store_nbr values: {frame['store_nbr'].nunique():,}")
    print(f"Unique family values: {frame['family'].nunique():,}")
    return set(dates.dropna().unique())


def count_duplicate_keys() -> int:
    """Count repeated date/store/family keys using a compact DataFrame state."""
    train_path = RAW_DATA_DIR / DATASET_FILES["train"]
    seen_keys = pd.DataFrame(columns=TRAIN_TEST_KEY_COLUMNS)
    duplicate_count = 0

    for chunk in pd.read_csv(
        train_path,
        usecols=TRAIN_TEST_KEY_COLUMNS,
        chunksize=CHUNK_SIZE,
    ):
        combined = pd.concat([seen_keys, chunk], ignore_index=True)
        duplicate_mask = combined.duplicated(
            subset=TRAIN_TEST_KEY_COLUMNS,
            keep="first",
        )
        duplicate_count += int(duplicate_mask.sum())
        seen_keys = combined.loc[
            ~duplicate_mask, TRAIN_TEST_KEY_COLUMNS
        ].copy()

    return duplicate_count


def calculate_median_sales() -> float:
    """Calculate the exact sales median from a targeted single-column read."""
    train_path = RAW_DATA_DIR / DATASET_FILES["train"]
    sales = pd.read_csv(train_path, usecols=["sales"])["sales"]
    return float(sales.median())


def profile_train_in_chunks() -> set[pd.Timestamp]:
    """Profile train.csv in chunks and return its unique dates."""
    train_path = RAW_DATA_DIR / DATASET_FILES["train"]
    date_values: set[pd.Timestamp] = set()
    store_values: set[object] = set()
    family_values: set[object] = set()
    combinations: set[tuple[object, object]] = set()
    row_count = 0
    column_names: list[str] = []
    dtypes: dict[str, str] = {}
    missing_counts: dict[str, int] = {}
    sales_count = 0
    sales_sum = 0.0
    sales_min = float("inf")
    sales_max = float("-inf")

    for chunk in pd.read_csv(train_path, chunksize=CHUNK_SIZE):
        if not column_names:
            column_names = list(chunk.columns)
            dtypes = {column: str(dtype) for column, dtype in chunk.dtypes.items()}
            missing_counts = {column: 0 for column in column_names}

        row_count += len(chunk)
        for column in column_names:
            missing_counts[column] += int(chunk[column].isna().sum())

        dates = pd.to_datetime(chunk["date"], errors="coerce")
        date_values.update(dates.dropna().unique())
        store_values.update(chunk["store_nbr"].dropna().unique())
        family_values.update(chunk["family"].dropna().unique())
        combinations.update(
            chunk[["store_nbr", "family"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )

        sales = chunk["sales"].dropna()
        if not sales.empty:
            sales_count += len(sales)
            sales_sum += float(sales.sum())
            sales_min = min(sales_min, float(sales.min()))
            sales_max = max(sales_max, float(sales.max()))

    print(f"\n{'=' * 16} train {'=' * 16}")
    print(f"Rows: {row_count:,}")
    print(f"Columns: {len(column_names):,}")
    print("Column details:")
    for column in column_names:
        print(
            f"  {column}: dtype={dtypes[column]}, "
            f"missing={missing_counts[column]:,}"
        )

    ordered_dates = sorted(date_values)
    print(f"Minimum date: {ordered_dates[0] if ordered_dates else 'N/A'}")
    print(f"Maximum date: {ordered_dates[-1] if ordered_dates else 'N/A'}")
    print(f"Unique store_nbr values: {len(store_values):,}")
    print(f"Unique family values: {len(family_values):,}")
    print(f"Total sales: {sales_sum:,.6f}")
    mean_sales = sales_sum / sales_count if sales_count else float("nan")
    median_sales = calculate_median_sales() if sales_count else float("nan")
    print(f"Mean sales: {mean_sales:,.6f}")
    print(f"Median sales: {median_sales:,.6f}")
    print(f"Minimum sales: {sales_min if sales_count else float('nan'):,.6f}")
    print(f"Maximum sales: {sales_max if sales_count else float('nan'):,.6f}")
    print(f"Unique dates: {len(date_values):,}")
    print(f"Unique store/family combinations: {len(combinations):,}")
    duplicate_count = count_duplicate_keys()
    print(
        "Duplicate date + store_nbr + family combinations: "
        f"{duplicate_count:,}"
    )

    test_dates = profile_test()
    overlapping_dates = date_values.intersection(test_dates)
    print("\n================ Train/test comparison ================")
    print(f"Train/test overlapping dates: {len(overlapping_dates):,}")
    if overlapping_dates:
        print(
            "Overlapping date range: "
            f"{min(overlapping_dates)} to {max(overlapping_dates)}"
        )

    return date_values


def profile_test() -> set[pd.Timestamp]:
    """Load test.csv, print its profile, and return its date values."""
    frame = load_dataset("test")
    print_dataset_profile("test", frame)
    return print_date_and_cardinality_profile(frame)


def main() -> None:
    """Print profiles for all raw datasets and train/test comparisons."""
    profile_train_in_chunks()
    for name in ("stores", "oil", "holidays_events", "transactions"):
        frame = load_dataset(name)
        print_dataset_profile(name, frame)


if __name__ == "__main__":
    main()
