"""Forecast-time-known calendar and exogenous feature preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

OIL_COLUMNS = {"date", "dcoilwtico"}
HOLIDAY_COLUMNS = {"date", "type", "locale", "locale_name", "transferred"}
STORE_COLUMNS = ["store_nbr", "city", "state", "type", "cluster"]


def prepare_oil_prices(oil: pd.DataFrame) -> pd.Series:
    """Create a calendar-aligned causal oil price lookup.

    Missing quotes carry the latest previously known quote forward. A leading
    missing quote remains missing; this function never backward-fills it.
    """
    missing = sorted(OIL_COLUMNS - set(oil.columns))
    if missing:
        raise ValueError(f"oil data is missing required columns: {missing}")
    prices = oil.loc[:, ["date", "dcoilwtico"]].copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="raise")
    if prices["date"].duplicated().any():
        raise ValueError("oil data must have at most one row per date")
    prices = prices.set_index("date")["dcoilwtico"].astype(float).sort_index()
    full_dates = pd.date_range(prices.index.min(), prices.index.max(), freq="D")
    return prices.reindex(full_dates).ffill()


def holiday_value_for_store(
    date: pd.Timestamp,
    store: pd.Series,
    holidays: pd.DataFrame,
) -> int:
    """Return the existing simple, scoped holiday/event indicator.

    Non-transferred Holiday, Additional, Event, Transfer, and Bridge rows are
    coded +1. Work Day is coded -1. Transferred source dates are omitted; the
    corresponding Transfer date remains eligible. This compact indicator does
    not distinguish event-specific sales effects.
    """
    same_date = holidays.loc[holidays["date"].eq(date)]
    locale_match = (
        same_date["locale"].eq("National")
        | (
            same_date["locale"].eq("Regional")
            & same_date["locale_name"].eq(store["state"])
        )
        | (same_date["locale"].eq("Local") & same_date["locale_name"].eq(store["city"]))
    )
    applicable = same_date.loc[locale_match & ~same_date["transferred"].astype(bool)]
    types = set(applicable["type"])
    if "Work Day" in types:
        return -1
    if types.intersection({"Holiday", "Additional", "Event", "Transfer", "Bridge"}):
        return 1
    return 0


def add_known_features(
    data: pd.DataFrame,
    stores: pd.DataFrame,
    oil_prices: pd.Series,
    holidays_events: pd.DataFrame,
) -> pd.DataFrame:
    """Add oil, holiday, payday, and store metadata to supplied rows.

    Rows are not synthesized, and target values are neither read nor changed.
    Oil values are aligned by exact calendar date. Holiday scope and transfer
    behavior match the existing SARIMAX evaluation helper.
    """
    required = {"date", "store_nbr"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    missing_stores = sorted(set(STORE_COLUMNS) - set(stores.columns))
    if missing_stores:
        raise ValueError(f"stores is missing required columns: {missing_stores}")
    missing_holidays = sorted(HOLIDAY_COLUMNS - set(holidays_events.columns))
    if missing_holidays:
        raise ValueError(
            f"holidays_events is missing required columns: {missing_holidays}"
        )

    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError("data.date must not contain missing values")
    if set(STORE_COLUMNS[1:]).intersection(result.columns):
        raise ValueError("data already contains store metadata columns")
    metadata = stores.loc[:, STORE_COLUMNS].copy()
    if metadata["store_nbr"].duplicated().any():
        raise ValueError("stores.store_nbr values must be unique")
    if set(result["store_nbr"]) - set(metadata["store_nbr"]):
        raise ValueError("stores metadata is missing a store_nbr in data")
    result = result.merge(metadata, on="store_nbr", how="left", validate="many_to_one")
    result["oil_price"] = oil_prices.reindex(
        pd.DatetimeIndex(result["date"])
    ).to_numpy()
    result["payday"] = (
        result["date"].dt.day.eq(15) | result["date"].dt.is_month_end
    ).astype("int8")

    holidays = holidays_events.loc[:, sorted(HOLIDAY_COLUMNS)].copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise")
    if holidays["date"].dt.normalize().ne(holidays["date"]).any():
        raise ValueError("holidays_events.date must contain calendar-day dates")
    store_lookup = metadata.set_index("store_nbr")
    holiday_lookup: dict[tuple[pd.Timestamp, object], int] = {}
    for date, day_events in holidays.groupby("date", sort=False):
        for store_nbr, store in store_lookup.iterrows():
            holiday_lookup[(pd.Timestamp(date), store_nbr)] = holiday_value_for_store(
                pd.Timestamp(date), store, day_events
            )
    keys = pd.MultiIndex.from_arrays(
        [result["date"], result["store_nbr"]], names=["date", "store_nbr"]
    )
    indexed_holidays = pd.Series(holiday_lookup)
    result["holiday_indicator"] = (
        indexed_holidays.reindex(keys).fillna(0).to_numpy(dtype=np.int8)
    )
    return result
