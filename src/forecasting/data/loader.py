"""Load the raw Favorita CSV datasets without modifying their contents."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATASET_FILES = {
    "train": "train.csv",
    "test": "test.csv",
    "stores": "stores.csv",
    "oil": "oil.csv",
    "holidays_events": "holidays_events.csv",
    "transactions": "transactions.csv",
}


def load_dataset(name: str) -> pd.DataFrame:
    """Load one supported raw CSV dataset by its short name.

    Args:
        name: Dataset name, such as ``"train"`` or ``"holidays_events"``.

    Returns:
        The CSV contents as a pandas DataFrame, as read from disk.

    Raises:
        ValueError: If ``name`` is not one of the supported datasets.
        FileNotFoundError: If the requested CSV file does not exist.
    """
    try:
        filename = DATASET_FILES[name]
    except KeyError as exc:
        supported = ", ".join(DATASET_FILES)
        raise ValueError(
            f"Unknown dataset {name!r}. Choose one of: {supported}."
        ) from exc

    csv_path = RAW_DATA_DIR / filename
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Required dataset file not found: {csv_path}. "
            "Place the Kaggle CSV in data/raw/."
        )

    return pd.read_csv(csv_path)


def load_train() -> pd.DataFrame:
    """Load the raw training dataset."""
    return load_dataset("train")


def load_test() -> pd.DataFrame:
    """Load the raw test dataset."""
    return load_dataset("test")


def load_stores() -> pd.DataFrame:
    """Load the raw store metadata dataset."""
    return load_dataset("stores")


def load_oil() -> pd.DataFrame:
    """Load the raw oil price dataset."""
    return load_dataset("oil")


def load_holidays_events() -> pd.DataFrame:
    """Load the raw holidays and events dataset."""
    return load_dataset("holidays_events")


def load_transactions() -> pd.DataFrame:
    """Load the raw transactions dataset."""
    return load_dataset("transactions")
