"""Backtest global LightGBM and CatBoost on representative series."""

from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

backtesting = import_module("forecasting.backtesting")
data_loader = import_module("forecasting.data.loader")
metrics = import_module("forecasting.evaluation")
feature_known = import_module("forecasting.features.known")
models = import_module("forecasting.models")
target_imputation = import_module("forecasting.data.imputation")

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
MODEL_FACTORIES = {
    "LightGBM": models.LightGBMModel,
    "CatBoost": models.CatBoostModel,
}

HISTORICAL_RESULTS = [
    {
        "model": "Naive",
        "mean_rmsle": 0.4336,
        "median_rmsle": 0.3569,
        "mean_mae": 1940.1267,
        "mean_rmse": 2351.0481,
        "coverage": "15 folds",
    },
    {
        "model": "Seasonal Naive (7-day)",
        "mean_rmsle": 0.1697,
        "median_rmsle": 0.1842,
        "mean_mae": 629.2666,
        "mean_rmse": 832.8903,
        "coverage": "15 folds",
    },
    {
        "model": "ARIMA(1,1,1)",
        "mean_rmsle": 0.3009,
        "median_rmsle": 0.2977,
        "mean_mae": 1221.6001,
        "mean_rmse": 1514.4389,
        "coverage": "15 folds",
    },
    {
        "model": "SARIMA(1,1,1)(1,0,1,7)",
        "mean_rmsle": 0.1889,
        "median_rmsle": 0.1778,
        "mean_mae": 881.6867,
        "mean_rmse": 1094.2058,
        "coverage": "14 successful / 15 folds",
    },
    {
        "model": "SARIMAX(1,1,1)(1,0,1,7), Powell",
        "mean_rmsle": 0.1386,
        "median_rmsle": 0.1388,
        "mean_mae": 538.3786,
        "mean_rmse": 719.4424,
        "coverage": "15 folds, latest Powell run",
    },
]


def prepare_validation(fold: backtesting.BacktestFold) -> pd.DataFrame:
    """Pair fold-held-out targets with keys, outside all model inputs."""
    validation = fold.validation_features.reset_index()
    validation["actual"] = fold.validation_target.to_numpy()
    selected = validation.loc[
        validation.set_index(SERIES_COLUMNS).index.isin(SERIES_CONFIG)
    ]
    counts = selected.groupby(SERIES_COLUMNS).size()
    if (
        set(counts.index.tolist()) != set(SERIES_CONFIG)
        or not counts.eq(FORECAST_HORIZON).all()
    ):
        raise ValueError(
            "Each representative series must have a complete 16-day validation "
            f"horizon; observed counts: {counts.to_dict()}"
        )
    return validation


def prepare_training_features(
    fold: backtesting.BacktestFold,
    stores: pd.DataFrame,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Impute targets within this fold, then attach known covariates."""
    train_rows = fold.train_data.reset_index().loc[
        :, [*KEY_COLUMNS, "sales", "onpromotion"]
    ]
    imputed = target_imputation.impute_training_sales_for_modeling(
        train_rows,
        stores,
        holidays,
        train_end=fold.train_end,
        complete_daily_grid=True,
    )
    return feature_known.add_known_features(imputed, stores, oil_prices, holidays)


def evaluate_fold(
    fold: backtesting.BacktestFold,
    validation: pd.DataFrame,
    training_features: pd.DataFrame,
    stores: pd.DataFrame,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> list[dict[str, object]]:
    """Fit each global model once and score the five selected series."""
    selected_mask = validation.set_index(SERIES_COLUMNS).index.isin(SERIES_CONFIG)
    selected_validation = validation.loc[selected_mask].copy()
    future_known = feature_known.add_known_features(
        selected_validation.drop(columns="actual"),
        stores,
        oil_prices,
        holidays,
    )
    actuals = selected_validation.loc[:, [*KEY_COLUMNS, "actual"]]
    rows: list[dict[str, object]] = []

    for model_name, model_factory in MODEL_FACTORIES.items():
        model = model_factory()
        fit_started = time.perf_counter()
        try:
            model.fit(training_features)
        except Exception as exc:
            fit_seconds = time.perf_counter() - fit_started
            for store_nbr, family in SERIES_CONFIG:
                rows.append(
                    {
                        "store_nbr": store_nbr,
                        "family": family,
                        "forecast_origin": fold.forecast_origin.date(),
                        "model": model_name,
                        "rmsle": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "fit_seconds": fit_seconds,
                        "prediction_seconds": np.nan,
                        "status": "fit_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        fit_seconds = time.perf_counter() - fit_started

        predict_started = time.perf_counter()
        try:
            predictions = model.predict(future_known)
        except Exception as exc:
            predict_seconds = time.perf_counter() - predict_started
            for store_nbr, family in SERIES_CONFIG:
                rows.append(
                    {
                        "store_nbr": store_nbr,
                        "family": family,
                        "forecast_origin": fold.forecast_origin.date(),
                        "model": model_name,
                        "rmsle": np.nan,
                        "mae": np.nan,
                        "rmse": np.nan,
                        "fit_seconds": fit_seconds,
                        "prediction_seconds": predict_seconds,
                        "status": "prediction_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        predict_seconds = time.perf_counter() - predict_started
        scored = predictions.merge(
            actuals,
            on=KEY_COLUMNS,
            how="inner",
            validate="one_to_one",
        )
        counts = scored.groupby(SERIES_COLUMNS).size()
        if (
            set(counts.index.tolist()) != set(SERIES_CONFIG)
            or not counts.eq(FORECAST_HORIZON).all()
        ):
            raise ValueError("ML predictions did not align to all validation targets")

        for store_nbr, family in SERIES_CONFIG:
            series_rows = scored.loc[
                scored["store_nbr"].eq(store_nbr) & scored["family"].eq(family)
            ]
            rows.append(
                {
                    "store_nbr": store_nbr,
                    "family": family,
                    "forecast_origin": fold.forecast_origin.date(),
                    "model": model_name,
                    "rmsle": metrics.rmsle(
                        series_rows["actual"], series_rows["prediction"]
                    ),
                    "mae": metrics.mae(
                        series_rows["actual"], series_rows["prediction"]
                    ),
                    "rmse": metrics.rmse(
                        series_rows["actual"], series_rows["prediction"]
                    ),
                    "fit_seconds": fit_seconds,
                    "prediction_seconds": predict_seconds,
                    "status": "success",
                    "error": None,
                }
            )
    return rows


def print_comparison(results: pd.DataFrame) -> None:
    """Print per-fold metrics and compare with recorded classical results."""
    print("Per-series, per-fold global ML results")
    print(
        results.loc[
            :,
            [
                "store_nbr",
                "family",
                "forecast_origin",
                "model",
                "rmsle",
                "mae",
                "rmse",
                "fit_seconds",
                "status",
            ],
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )
    failed = results.loc[~results["status"].eq("success")]
    print("\nFailed fits/predictions")
    print("None" if failed.empty else failed.to_string(index=False))

    successful = results.loc[results["status"].eq("success")]
    aggregate = successful.groupby("model", as_index=False).agg(
        total_fits=("rmsle", "size"),
        mean_rmsle=("rmsle", "mean"),
        median_rmsle=("rmsle", "median"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
        mean_fit_seconds=("fit_seconds", "mean"),
    )
    aggregate["failed_fits_or_predictions"] = [
        int(((results["model"] == model_name) & ~results["status"].eq("success")).sum())
        for model_name in aggregate["model"]
    ]
    historical = pd.DataFrame(HISTORICAL_RESULTS)
    historical["total_fits"] = historical["coverage"]
    historical["failed_fits_or_predictions"] = "see coverage"
    historical["mean_fit_seconds"] = np.nan
    comparison = pd.concat(
        [
            historical.loc[:, aggregate.columns],
            aggregate.loc[:, aggregate.columns],
        ],
        ignore_index=True,
    )
    print("\nAggregate comparison (classical results are previously recorded)")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Run both global ML models on the established three rolling folds."""
    raw_train = data_loader.load_train()
    raw_train["date"] = pd.to_datetime(raw_train["date"], errors="raise")
    panel = raw_train.loc[:, [*KEY_COLUMNS, "sales", "onpromotion"]].copy()
    selected_series = pd.MultiIndex.from_frame(panel[SERIES_COLUMNS]).isin(
        pd.MultiIndex.from_tuples(SERIES_CONFIG, names=SERIES_COLUMNS)
    )
    panel = panel.loc[selected_series].copy()
    folds = backtesting.generate_rolling_folds(
        panel.set_index("date").sort_index(),
        forecast_origins=FORECAST_ORIGINS,
        horizon=FORECAST_HORIZON,
    )
    if len(folds) != len(FORECAST_ORIGINS):
        raise ValueError(
            f"Expected {len(FORECAST_ORIGINS)} complete folds, generated {len(folds)}"
        )

    stores = data_loader.load_stores()
    oil_prices = feature_known.prepare_oil_prices(data_loader.load_oil())
    holidays = data_loader.load_holidays_events().copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise")
    results: list[dict[str, object]] = []
    for fold in folds:
        print(
            f"Preparing fold with forecast origin {fold.forecast_origin.date()}...",
            flush=True,
        )
        validation = prepare_validation(fold)
        training_features = prepare_training_features(
            fold,
            stores,
            oil_prices,
            holidays,
        )
        results.extend(
            evaluate_fold(
                fold,
                validation,
                training_features,
                stores,
                oil_prices,
                holidays,
            )
        )
        print(f"Completed forecast origin {fold.forecast_origin.date()}.", flush=True)
    print_comparison(pd.DataFrame(results))


if __name__ == "__main__":
    main()
