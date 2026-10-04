"""Evaluate fixed SARIMAX with known-date exogenous variables."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from forecasting.features.known import (  # noqa: E402
    holiday_value_for_store,
    prepare_oil_prices,
)

backtesting = import_module("forecasting.backtesting")
data_loader = import_module("forecasting.data.loader")
metrics = import_module("forecasting.evaluation")
models = import_module("forecasting.models")

SERIES_CONFIG = [
    (44, "GROCERY I"),
    (1, "GROCERY I"),
    (10, "BEVERAGES"),
    (1, "PRODUCE"),
    (45, "BEVERAGES"),
]
FORECAST_ORIGINS = pd.to_datetime(["2017-06-28", "2017-07-14", "2017-07-30"])
FORECAST_HORIZON = 16
SERIES_COLUMNS = ["store_nbr", "family"]
KEY_COLUMNS = ["date", *SERIES_COLUMNS]
MODEL_NAME = "SARIMAX(1,1,1)(1,0,1,7)"


def build_exogenous_rows(
    rows: pd.DataFrame,
    store: pd.Series,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Align known exogenous inputs to the exact rows and dates supplied."""
    result = rows.loc[:, [*KEY_COLUMNS, "onpromotion"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    dates = pd.DatetimeIndex(result["date"])
    result["oil_price"] = oil_prices.reindex(dates).to_numpy()
    result["holiday_indicator"] = [
        holiday_value_for_store(date, store, holidays) for date in dates
    ]
    result["payday"] = (
        result["date"].dt.day.eq(15) | result["date"].dt.is_month_end
    ).astype("int8")
    return result


def prepare_validation(fold: backtesting.BacktestFold) -> pd.DataFrame:
    """Pair held-out targets with keys for scoring, outside model input."""
    validation = fold.validation_features.reset_index()
    validation["actual"] = fold.validation_target.to_numpy()
    counts = validation.groupby(SERIES_COLUMNS).size()
    if (
        set(counts.index.tolist()) != set(SERIES_CONFIG)
        or not counts.eq(FORECAST_HORIZON).all()
    ):
        raise ValueError(
            "Each configured series must have all 16 validation observations; "
            f"observed counts: {counts.to_dict()}"
        )
    return validation


def score_series(
    fold: backtesting.BacktestFold,
    validation: pd.DataFrame,
    store_nbr: int,
    family: str,
    store_lookup: pd.DataFrame,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> dict[str, object]:
    """Fit and score one series, recording fit-time numerical failures."""
    series_filter = fold.train_data["store_nbr"].eq(store_nbr) & fold.train_data[
        "family"
    ].eq(family)
    train_rows = fold.train_data.loc[series_filter].reset_index()
    future_filter = validation["store_nbr"].eq(store_nbr) & validation["family"].eq(
        family
    )
    validation_rows = validation.loc[future_filter]
    store = store_lookup.loc[store_nbr]

    train_data = build_exogenous_rows(train_rows, store, oil_prices, holidays)
    train_data["sales"] = train_rows["sales"].to_numpy()
    future_data = build_exogenous_rows(
        validation_rows.drop(columns="actual"), store, oil_prices, holidays
    )

    result = {
        "store_nbr": store_nbr,
        "family": family,
        "forecast_origin": fold.forecast_origin.date(),
        "validation_start": fold.validation_start.date(),
        "validation_end": fold.validation_end.date(),
        "model": MODEL_NAME,
    }
    model = models.SARIMAXModel()
    try:
        model.fit(train_data)
    except RuntimeError as exc:
        return {
            **result,
            "rmsle": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "status": "fit_failed",
            "error": str(exc),
        }

    predictions = model.predict(future_data)
    actuals = validation_rows.loc[:, [*KEY_COLUMNS, "actual"]]
    scored = predictions.merge(
        actuals,
        on=KEY_COLUMNS,
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != FORECAST_HORIZON:
        raise ValueError(
            f"Prediction and actual rows did not align for {store_nbr}, {family!r}"
        )
    return {
        **result,
        "rmsle": metrics.rmsle(scored["actual"], scored["prediction"]),
        "mae": metrics.mae(scored["actual"], scored["prediction"]),
        "rmse": metrics.rmse(scored["actual"], scored["prediction"]),
        "status": "success",
        "error": None,
    }


def report_results(results: pd.DataFrame) -> None:
    """Print per-fold results, failed fits, and success-only aggregates."""
    successful = results.loc[results["status"].eq("success")]
    failed = results.loc[results["status"].eq("fit_failed")]
    result_columns = [
        "store_nbr",
        "family",
        "forecast_origin",
        "model",
        "rmsle",
        "mae",
        "rmse",
    ]
    print("Successful per-series, per-fold SARIMAX metrics")
    print(
        successful.loc[:, result_columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )
    print("\nFailed SARIMAX fits")
    if failed.empty:
        print("None")
    else:
        print(
            failed.loc[
                :,
                ["store_nbr", "family", "forecast_origin", "error"],
            ].to_string(index=False)
        )

    summary = {
        "mean_rmsle": successful["rmsle"].mean(),
        "median_rmsle": successful["rmsle"].median(),
        "mean_mae": successful["mae"].mean(),
        "mean_rmse": successful["rmse"].mean(),
    }
    print("\nAggregate SARIMAX summary (successful fits only)")
    print(f"Successful fits: {len(successful)}")
    print(f"Failed fits: {len(failed)}")
    for name, value in summary.items():
        print(f"{name}: {value:.4f}" if pd.notna(value) else f"{name}: NaN")


def main() -> None:
    """Evaluate SARIMAX on five representative series and three fixed folds."""
    raw_train = data_loader.load_train()
    oil_prices = prepare_oil_prices(data_loader.load_oil())
    holidays = data_loader.load_holidays_events().copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise")
    stores = data_loader.load_stores().set_index("store_nbr")

    series_mask = pd.Series(False, index=raw_train.index)
    for store_nbr, family in SERIES_CONFIG:
        series_mask |= raw_train["store_nbr"].eq(store_nbr) & raw_train["family"].eq(
            family
        )
    panel = raw_train.loc[series_mask, [*KEY_COLUMNS, "sales", "onpromotion"]].copy()
    if panel.empty:
        raise ValueError("None of the configured store-family series were found")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    folds = backtesting.generate_rolling_folds(
        panel.set_index("date").sort_index(),
        forecast_origins=FORECAST_ORIGINS,
        horizon=FORECAST_HORIZON,
    )
    if len(folds) != len(FORECAST_ORIGINS):
        raise ValueError(
            f"Expected {len(FORECAST_ORIGINS)} complete folds, generated {len(folds)}"
        )

    rows = []
    for fold in folds:
        validation = prepare_validation(fold)
        for store_nbr, family in SERIES_CONFIG:
            rows.append(
                score_series(
                    fold,
                    validation,
                    store_nbr,
                    family,
                    stores,
                    oil_prices,
                    holidays,
                )
            )
    report_results(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
