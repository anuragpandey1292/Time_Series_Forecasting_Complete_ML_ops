"""Unified representative-scope training and rolling-origin evaluation entry point."""

from __future__ import annotations

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

backtesting = import_module("forecasting.backtesting")
data_loader = import_module("forecasting.data.loader")
raw_validation = import_module("forecasting.data.validation")
feature_known = import_module("forecasting.features.known")
metrics = import_module("forecasting.evaluation")
models = import_module("forecasting.models")
target_imputation = import_module("forecasting.data.imputation")
tracking = import_module("forecasting.tracking")

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

MODEL_REGISTRY: dict[str, type[Any]] = {
    "naive": models.NaiveModel,
    "seasonal_naive": models.SeasonalNaiveModel,
    "arima": models.ARIMAModel,
    "sarima": models.SARIMAModel,
    "sarimax": models.SARIMAXModel,
    "lightgbm": models.LightGBMModel,
    "catboost": models.CatBoostModel,
}


def create_model(model_name: str) -> Any:
    """Create the configured implementation for a CLI model name."""
    try:
        model_type = MODEL_REGISTRY[model_name]
    except KeyError as exc:
        supported = ", ".join(MODEL_REGISTRY)
        raise ValueError(
            f"Unsupported model {model_name!r}; choose one of: {supported}"
        ) from exc
    if model_name == "seasonal_naive":
        return model_type(seasonal_period=7)
    return model_type()


def select_panel(raw_train: pd.DataFrame) -> pd.DataFrame:
    """Select the five representative series without changing raw data."""
    mask = pd.Series(False, index=raw_train.index)
    for store_nbr, family in SERIES_CONFIG:
        mask |= raw_train["store_nbr"].eq(store_nbr) & raw_train["family"].eq(family)
    panel = raw_train.loc[mask].copy()
    if panel.empty:
        raise ValueError("None of the representative series were found")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    return panel


def prepare_validation(fold: backtesting.BacktestFold) -> pd.DataFrame:
    """Return fold targets for scoring, outside all model inputs."""
    validation = fold.validation_features.reset_index()
    validation["actual"] = fold.validation_target.to_numpy()
    counts = validation.groupby(SERIES_COLUMNS).size()
    if (
        set(counts.index.tolist()) != set(SERIES_CONFIG)
        or not counts.eq(FORECAST_HORIZON).all()
    ):
        raise ValueError("Every representative series needs a complete validation fold")
    return validation


def prepare_global_training(
    fold: backtesting.BacktestFold,
    stores: pd.DataFrame,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare one fold's leakage-safe global training frame."""
    rows = fold.train_data.reset_index().loc[:, [*KEY_COLUMNS, "sales", "onpromotion"]]
    prepared = target_imputation.impute_training_sales_for_modeling(
        rows,
        stores,
        holidays,
        train_end=fold.train_end,
        complete_daily_grid=True,
    )
    return feature_known.add_known_features(prepared, stores, oil_prices, holidays)


def prepare_sarimax_rows(
    rows: pd.DataFrame,
    store: pd.Series,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the known exogenous inputs expected by SARIMAX."""
    result = rows.loc[:, [*KEY_COLUMNS, "onpromotion"]].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    dates = pd.DatetimeIndex(result["date"])
    result["oil_price"] = oil_prices.reindex(dates).to_numpy()
    result["holiday_indicator"] = [
        feature_known.holiday_value_for_store(date, store, holidays) for date in dates
    ]
    result["payday"] = (
        result["date"].dt.day.eq(15) | result["date"].dt.is_month_end
    ).astype("int8")
    return result


def score_one(
    model_name: str,
    fold: backtesting.BacktestFold,
    validation: pd.DataFrame,
    store_nbr: int,
    family: str,
    stores: pd.DataFrame,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> dict[str, object]:
    """Fit and score one classical model-series fold."""
    train_mask = fold.train_data["store_nbr"].eq(store_nbr) & fold.train_data[
        "family"
    ].eq(family)
    train_rows = fold.train_data.loc[train_mask].reset_index()
    validation_rows = validation.loc[
        validation["store_nbr"].eq(store_nbr) & validation["family"].eq(family)
    ]
    if model_name == "sarimax":
        store = stores.set_index("store_nbr").loc[store_nbr]
        train_data = prepare_sarimax_rows(train_rows, store, oil_prices, holidays)
        train_data["sales"] = train_rows["sales"].to_numpy()
    else:
        train_data = train_rows
    model = create_model(model_name)
    result = {
        "store_nbr": store_nbr,
        "family": family,
        "forecast_origin": fold.forecast_origin.date(),
        "model": model_name,
    }
    try:
        model.fit(train_data)
        future = validation_rows.drop(columns="actual")
        if model_name == "sarimax":
            store = stores.set_index("store_nbr").loc[store_nbr]
            future = prepare_sarimax_rows(future, store, oil_prices, holidays)
        predictions = model.predict(future)
        actuals = validation_rows.loc[:, [*KEY_COLUMNS, "actual"]]
        scored = predictions.merge(
            actuals, on=KEY_COLUMNS, how="inner", validate="one_to_one"
        )
        if len(scored) != FORECAST_HORIZON:
            raise ValueError("Prediction and validation rows did not align")
        return {
            **result,
            "rmsle": metrics.rmsle(scored["actual"], scored["prediction"]),
            "mae": metrics.mae(scored["actual"], scored["prediction"]),
            "rmse": metrics.rmse(scored["actual"], scored["prediction"]),
            "status": "success",
            "error": None,
        }
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        return {
            **result,
            "rmsle": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "status": "fit_or_prediction_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def model_parameters(model_name: str) -> dict[str, object]:
    """Return explicit fixed parameters for tracking."""
    return {
        "naive": {"strategy": "latest observed value"},
        "seasonal_naive": {"seasonal_lag": 7},
        "arima": {"order": (1, 1, 1)},
        "sarima": {"order": (1, 1, 1), "seasonal_order": (1, 0, 1, 7)},
        "sarimax": {
            "order": (1, 1, 1),
            "seasonal_order": (1, 0, 1, 7),
            "optimizer": "powell",
            "maxiter": 200,
            "exogenous_features": "onpromotion,oil_price,holiday_indicator,payday",
        },
    }.get(model_name, {})


def run_pipeline(
    model_name: str, mlflow_enabled: bool = True
) -> tuple[pd.DataFrame, tracking.TrackingOutcome]:
    """Run the representative rolling-origin pipeline for one model."""
    started = time.perf_counter()
    raw_train = data_loader.load_train()
    raw_validation.validate_datasets(
        train=raw_train,
        test=data_loader.load_test(),
        stores=data_loader.load_stores(),
        oil=data_loader.load_oil(),
        holidays_events=data_loader.load_holidays_events(),
        transactions=data_loader.load_transactions(),
    )
    panel = select_panel(raw_train)
    folds = backtesting.generate_rolling_folds(
        panel.set_index("date").sort_index(),
        forecast_origins=FORECAST_ORIGINS,
        horizon=FORECAST_HORIZON,
    )
    stores = data_loader.load_stores()
    oil_prices = feature_known.prepare_oil_prices(data_loader.load_oil())
    holidays = data_loader.load_holidays_events().copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise")
    rows: list[dict[str, object]] = []
    for fold in folds:
        validation = prepare_validation(fold)
        if model_name in {"lightgbm", "catboost"}:
            training = prepare_global_training(fold, stores, oil_prices, holidays)
            model = create_model(model_name)
            try:
                model.fit(training)
                selected = validation.loc[
                    validation.set_index(SERIES_COLUMNS).index.isin(SERIES_CONFIG)
                ].copy()
                future = feature_known.add_known_features(
                    selected.drop(columns="actual"), stores, oil_prices, holidays
                )
                predictions = model.predict(future)
                scored = predictions.merge(
                    selected[[*KEY_COLUMNS, "actual"]],
                    on=KEY_COLUMNS,
                    how="inner",
                    validate="one_to_one",
                )
                for store_nbr, family in SERIES_CONFIG:
                    series = scored.loc[
                        scored["store_nbr"].eq(store_nbr) & scored["family"].eq(family)
                    ]
                    rows.append(
                        {
                            "store_nbr": store_nbr,
                            "family": family,
                            "forecast_origin": fold.forecast_origin.date(),
                            "model": model_name,
                            "rmsle": metrics.rmsle(series.actual, series.prediction),
                            "mae": metrics.mae(series.actual, series.prediction),
                            "rmse": metrics.rmse(series.actual, series.prediction),
                            "status": "success",
                            "error": None,
                        }
                    )
            except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                rows.extend(
                    {
                        "store_nbr": store_nbr,
                        "family": family,
                        "forecast_origin": fold.forecast_origin.date(),
                        "model": model_name,
                        "rmsle": float("nan"),
                        "mae": float("nan"),
                        "rmse": float("nan"),
                        "status": "fit_or_prediction_failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    for store_nbr, family in SERIES_CONFIG
                )
        else:
            rows.extend(
                score_one(
                    model_name,
                    fold,
                    validation,
                    store_nbr,
                    family,
                    stores,
                    oil_prices,
                    holidays,
                )
                for store_nbr, family in SERIES_CONFIG
            )
    results = pd.DataFrame(rows)
    outcome = tracking.log_evaluation_run(
        model_name=model_name,
        model=None,
        model_parameters=model_parameters(model_name),
        results=results,
        forecast_horizon=FORECAST_HORIZON,
        number_of_series=len(SERIES_CONFIG),
        number_of_folds=len(folds),
        forecast_origins=[fold.forecast_origin.strftime("%Y-%m-%d") for fold in folds],
        runtime_seconds=time.perf_counter() - started,
        project_root=PROJECT_ROOT,
        enabled=mlflow_enabled,
        tags={
            "model_family": (
                "global_ml" if model_name in {"lightgbm", "catboost"} else "classical"
            )
        },
        run_name=f"{model_name}-unified-training",
    )
    return results, outcome


def parse_args() -> argparse.Namespace:
    """Parse the small public command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY))
    parser.add_argument("--mlflow-enabled", default="true", choices=["true", "false"])
    return parser.parse_args()


def main() -> None:
    """Run one selected model and print its concise aggregate summary."""
    args = parse_args()
    results, outcome = run_pipeline(
        args.model, mlflow_enabled=args.mlflow_enabled == "true"
    )
    successful = results.loc[results["status"].eq("success")]
    failed_count = int((~results["status"].eq("success")).sum())
    mean_rmsle = successful["rmsle"].mean()
    median_rmsle = successful["rmsle"].median()
    mean_mae = successful["mae"].mean()
    mean_rmse = successful["rmse"].mean()
    print(f"Model: {args.model}")
    print(f"Folds: {len(successful)}")
    print(f"Failed fits: {failed_count}")
    print(f"Mean RMSLE: {mean_rmsle:.4f}")
    print(f"Median RMSLE: {median_rmsle:.4f}")
    print(f"Mean MAE: {mean_mae:.4f}")
    print(f"Mean RMSE: {mean_rmse:.4f}")
    print(f"MLflow run ID: {outcome.run_id or outcome.message or outcome.status}")


if __name__ == "__main__":
    main()
