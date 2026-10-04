"""Explicit, auditable imputation utilities for historical forecasting data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

GRAIN_COLUMNS = ["store_nbr", "family", "date"]
STORE_COLUMNS = ["store_nbr", "city", "state"]
HOLIDAY_COLUMNS = [
    "date",
    "type",
    "locale",
    "locale_name",
    "transferred",
]
CLOSURE_TYPES = {"Holiday", "Additional", "Transfer", "Bridge"}


def _as_day(value: object, name: str) -> pd.Timestamp:
    """Convert one date-like boundary to a normalized calendar day."""
    try:
        day = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid date") from exc
    if pd.isna(day):
        raise ValueError(f"{name} must be a valid date")
    day = day.normalize()
    return day


def _prepare_holidays(holidays_events: pd.DataFrame) -> pd.DataFrame:
    """Validate and copy the holiday fields used for scope and semantics."""
    missing = sorted(set(HOLIDAY_COLUMNS) - set(holidays_events.columns))
    if missing:
        raise ValueError(f"holidays_events is missing required columns: {missing}")
    holidays = holidays_events.loc[:, HOLIDAY_COLUMNS].copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise").dt.normalize()
    if (
        holidays[["date", "type", "locale", "locale_name", "transferred"]]
        .isna()
        .any()
        .any()
    ):
        raise ValueError(
            "holidays_events required fields must not contain missing values"
        )
    return holidays


def _is_applicable_closure(
    date: pd.Timestamp,
    store: pd.Series,
    holidays: pd.DataFrame,
) -> bool:
    """Return whether a date is a scoped, confirmed closure for one store.

    Transferred source dates are ignored; their separate ``Transfer`` event
    date is eligible. ``Holiday``, ``Additional``, ``Transfer``, and
    ``Bridge`` denote closures. ``Work Day`` denotes an open make-up day, and
    generic ``Event`` records are not assumed to close stores. If a Work Day
    and closure record overlap, the explicit Work Day status takes precedence.
    """
    same_date = holidays.loc[holidays["date"].eq(date)]
    applicable_scope = (
        same_date["locale"].eq("National")
        | (
            same_date["locale"].eq("Regional")
            & same_date["locale_name"].eq(store["state"])
        )
        | (same_date["locale"].eq("Local") & same_date["locale_name"].eq(store["city"]))
    )
    applicable = same_date.loc[
        applicable_scope & ~same_date["transferred"].astype(bool)
    ]
    types = set(applicable["type"])
    if "Work Day" in types:
        return False
    return bool(types.intersection(CLOSURE_TYPES))


def impute_training_sales(
    data: pd.DataFrame,
    stores: pd.DataFrame,
    holidays_events: pd.DataFrame,
    *,
    train_end: object,
    train_start: object | None = None,
    complete_daily_grid: bool = True,
) -> pd.DataFrame:
    """Impute sales with the generic forward-then-backward policy.

    ``train_end`` is mandatory and acts as a hard split boundary: any supplied
    row after it is rejected. When requested, each series is reindexed to a
    daily grid from its first supplied date (or ``train_start``) through that
    boundary. Applicable closure dates receive zero. Other gaps are forward-
    filled, then remaining leading gaps are backward-filled within that same
    series and training interval only.

    The original sales value is retained in ``sales_original``. Synthetic
    dates have a missing original value. Holiday rules are intentionally
    conservative: non-transferred Holiday, Additional, Transfer, and Bridge
    records are closures; Work Day overrides a closure; generic Event records
    alone do not establish store closure.

    This generic policy may use later values within the same training interval
    to fill leading gaps. Use :func:`impute_training_sales_for_modeling` for
    leakage-safe target preparation in forecasting and backtesting.
    """
    return _impute_training_sales(
        data,
        stores,
        holidays_events,
        train_end=train_end,
        train_start=train_start,
        complete_daily_grid=complete_daily_grid,
        allow_target_bfill=True,
    )


def impute_training_sales_for_modeling(
    data: pd.DataFrame,
    stores: pd.DataFrame,
    holidays_events: pd.DataFrame,
    *,
    train_end: object,
    train_start: object | None = None,
    complete_daily_grid: bool = True,
) -> pd.DataFrame:
    """Prepare target sales without backward fill or future-target leakage.

    Call separately for each rolling-origin fold, passing only that fold's
    training rows and its ``train_end``. Never impute the full historical panel
    once before splitting it. Confirmed closure dates are still set to zero;
    ordinary gaps use prior sales only, while leading gaps remain NaN and are
    audited with ``sales_imputation_method="unfilled"``.
    """
    return _impute_training_sales(
        data,
        stores,
        holidays_events,
        train_end=train_end,
        train_start=train_start,
        complete_daily_grid=complete_daily_grid,
        allow_target_bfill=False,
    )


def _impute_training_sales(
    data: pd.DataFrame,
    stores: pd.DataFrame,
    holidays_events: pd.DataFrame,
    *,
    train_end: object,
    train_start: object | None,
    complete_daily_grid: bool,
    allow_target_bfill: bool,
) -> pd.DataFrame:
    """Shared implementation for generic and modeling-safe target policies."""
    required = {*GRAIN_COLUMNS, "sales"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")
    missing_store_columns = sorted(set(STORE_COLUMNS) - set(stores.columns))
    if missing_store_columns:
        raise ValueError(f"stores is missing required columns: {missing_store_columns}")

    end_day = _as_day(train_end, "train_end")
    start_day = _as_day(train_start, "train_start") if train_start is not None else None
    source = data.copy()
    source["date"] = pd.to_datetime(source["date"], errors="raise")
    if source["date"].isna().any():
        raise ValueError("data.date must not contain missing values")
    if not (source["date"] == source["date"].dt.normalize()).all():
        raise ValueError("data.date must contain calendar-day dates")
    if (source["date"] > end_day).any():
        raise ValueError(
            "data contains dates after train_end; validation/test targets cannot "
            "be imputed as training data"
        )
    if start_day is not None and (source["date"] < start_day).any():
        raise ValueError("data contains dates before train_start")
    if source[["store_nbr", "family"]].isna().any().any():
        raise ValueError("data store_nbr and family must not contain missing values")
    if not pd.api.types.is_numeric_dtype(source["sales"]):
        raise ValueError("data.sales must be numeric")
    if source.duplicated(subset=GRAIN_COLUMNS).any():
        raise ValueError("data must be unique on date + store_nbr + family")

    store_metadata = stores.loc[:, STORE_COLUMNS].copy()
    if store_metadata["store_nbr"].duplicated().any():
        raise ValueError("stores.store_nbr values must be unique")
    store_metadata = store_metadata.set_index("store_nbr")
    unknown_stores = set(source["store_nbr"]) - set(store_metadata.index)
    if unknown_stores:
        raise ValueError(
            f"stores metadata is missing store_nbr values: {sorted(unknown_stores)}"
        )
    holidays = _prepare_holidays(holidays_events)

    series_frames = []
    for (store_nbr, family), group in source.groupby(
        ["store_nbr", "family"], sort=False
    ):
        group = group.sort_values("date", kind="stable").set_index("date")
        series_start = start_day or pd.Timestamp(group.index.min())
        if series_start > end_day:
            raise ValueError("train_start must not be after train_end")
        if complete_daily_grid:
            date_index = pd.date_range(series_start, end_day, freq="D")
            result = group.reindex(date_index)
            result.index.name = "date"
        else:
            result = group.copy()
        result["store_nbr"] = store_nbr
        result["family"] = family
        result = result.reset_index()
        result["sales_original"] = result["sales"].copy()
        result["sales_imputation_method"] = "observed"
        result["is_imputed"] = False

        missing_sales = result["sales"].isna()
        store = store_metadata.loc[store_nbr]
        closure_mask = pd.Series(False, index=result.index)
        for index in result.index[missing_sales]:
            date = result.at[index, "date"]
            closure_mask.at[index] = _is_applicable_closure(date, store, holidays)

        closure_indices = result.index[closure_mask]
        result.loc[closure_indices, "sales"] = 0
        result.loc[closure_indices, "sales_imputation_method"] = "holiday_zero"

        ordinary_missing = result["sales"].isna()
        forward_values = result["sales"].ffill()
        forward_indices = result.index[ordinary_missing & forward_values.notna()]
        result.loc[forward_indices, "sales"] = forward_values.loc[forward_indices]
        result.loc[forward_indices, "sales_imputation_method"] = "ffill"

        leading_missing = result["sales"].isna()
        if allow_target_bfill:
            backward_values = result["sales"].bfill()
            backward_indices = result.index[leading_missing & backward_values.notna()]
            result.loc[backward_indices, "sales"] = backward_values.loc[
                backward_indices
            ]
            result.loc[backward_indices, "sales_imputation_method"] = "bfill"
        else:
            result.loc[leading_missing, "sales_imputation_method"] = "unfilled"
        result["is_imputed"] = result["sales_imputation_method"].isin(
            {"holiday_zero", "ffill", "bfill"}
        )

        if allow_target_bfill and result["sales"].isna().any():
            raise ValueError(
                f"store {store_nbr}, family {family!r} has no observed sales "
                "value available for training-period imputation"
            )
        series_frames.append(result)

    if not series_frames:
        raise ValueError("data must contain at least one store-family series")
    output = pd.concat(series_frames, ignore_index=True)
    return output.sort_values(GRAIN_COLUMNS, kind="stable").reset_index(drop=True)


def impute_training_exogenous(
    data: pd.DataFrame,
    columns: Iterable[str],
    *,
    train_end: object,
    group_columns: Iterable[str] = (),
) -> pd.DataFrame:
    """Forward/backward-fill historical exogenous columns with audit flags.

    Every supplied date must be on or before the explicit ``train_end``.
    Filling is performed by date, optionally within caller-specified groups.
    Each requested column gets a ``<column>_was_imputed`` boolean, true only
    when a missing value was successfully filled. This helper does not operate
    on targets and never admits rows beyond the training boundary.
    """
    value_columns = list(columns)
    groups = list(group_columns)
    if not value_columns:
        raise ValueError("at least one exogenous column is required")
    if "sales" in value_columns:
        raise ValueError("sales is a target; use impute_training_sales instead")
    required = {"date", *value_columns, *groups}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"data is missing required columns: {missing}")

    end_day = _as_day(train_end, "train_end")
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    if result["date"].isna().any():
        raise ValueError("data.date must not contain missing values")
    if (result["date"] > end_day).any():
        raise ValueError(
            "data contains dates after train_end; exogenous training fill "
            "cannot cross a validation/test boundary"
        )
    for column in value_columns:
        if not pd.api.types.is_numeric_dtype(result[column]):
            raise ValueError(f"data.{column} must be numeric")

    sort_columns = [*groups, "date"]
    result = result.sort_values(sort_columns, kind="stable").reset_index(drop=True)
    groupby_columns: str | list[str] | None = groups or None
    for column in value_columns:
        was_missing = result[column].isna()
        if groupby_columns is None:
            filled = result[column].ffill().bfill()
        else:
            filled = result.groupby(groupby_columns, sort=False)[column].transform(
                lambda values: values.ffill().bfill()
            )
        result[column] = filled
        result[f"{column}_was_imputed"] = was_missing & filled.notna()
    return result
