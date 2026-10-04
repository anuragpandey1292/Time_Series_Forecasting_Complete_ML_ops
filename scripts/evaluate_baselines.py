"""Evaluate baseline forecasters across representative series and time folds."""

from __future__ import annotations

import sys
import time
from importlib import import_module
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

backtesting = import_module("forecasting.backtesting")
data_loader = import_module("forecasting.data.loader")
metrics = import_module("forecasting.evaluation")
models = import_module("forecasting.models")
tracking = import_module("forecasting.tracking")

SERIES_CONFIG = [
    (44, "GROCERY I"),
    (1, "GROCERY I"),
    (10, "BEVERAGES"),
    (1, "PRODUCE"),
    (45, "BEVERAGES"),
]
FORECAST_HORIZON = 16
FOLD_COUNT = 3
SERIES_COLUMNS = ["store_nbr", "family"]
KEY_COLUMNS = ["date", *SERIES_COLUMNS]


def make_fold_origins(latest_date: pd.Timestamp) -> list[pd.Timestamp]:
    """Return three consecutive, non-overlapping 16-day holdout origins."""
    return [
        latest_date - pd.Timedelta(days=FORECAST_HORIZON * offset)
        for offset in range(FOLD_COUNT, 0, -1)
    ]


def validate_fold_series_coverage(
    fold: backtesting.BacktestFold,
) -> pd.DataFrame:
    """Ensure each configured series has every date in the validation horizon."""
    validation = fold.validation_features.reset_index()
    validation["actual"] = fold.validation_target.to_numpy()
    counts = validation.groupby(SERIES_COLUMNS).size()
    expected_series = set(SERIES_CONFIG)
    actual_series = set(counts.index.tolist())
    if actual_series != expected_series or not counts.eq(FORECAST_HORIZON).all():
        raise ValueError(
            "Each configured series must have all 16 validation observations; "
            f"observed counts: {counts.to_dict()}"
        )
    return validation


def score_series_model(
    fold: backtesting.BacktestFold,
    validation: pd.DataFrame,
    store_nbr: int,
    family: str,
    model_name: str,
    model: models.ForecastModel,
) -> dict[str, object]:
    """Fit on one series' fold history and score its withheld horizon."""
    series_mask = (fold.train_data["store_nbr"] == store_nbr) & (
        fold.train_data["family"] == family
    )
    train_data = fold.train_data.loc[series_mask].reset_index()
    if train_data.empty:
        raise ValueError(f"No training history for store {store_nbr}, {family!r}")

    validation_mask = (validation["store_nbr"] == store_nbr) & (
        validation["family"] == family
    )
    series_validation = validation.loc[validation_mask]
    future_data = series_validation.drop(columns="actual").copy()
    model.fit(train_data)
    predictions = model.predict(future_data)

    actuals = series_validation.loc[:, [*KEY_COLUMNS, "actual"]]
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
        "store_nbr": store_nbr,
        "family": family,
        "forecast_origin": fold.forecast_origin.date(),
        "model": model_name,
        "rmsle": metrics.rmsle(scored["actual"], scored["prediction"]),
        "mae": metrics.mae(scored["actual"], scored["prediction"]),
        "rmse": metrics.rmse(scored["actual"], scored["prediction"]),
        "status": "success",
    }


def main() -> None:
    """Run the naive baselines on five series over three historical folds."""
    raw_train = data_loader.load_train()
    series_mask = pd.Series(False, index=raw_train.index)
    for store_nbr, family in SERIES_CONFIG:
        series_mask |= (raw_train["store_nbr"] == store_nbr) & (
            raw_train["family"] == family
        )

    panel = raw_train.loc[series_mask, [*KEY_COLUMNS, "sales"]].copy()
    if panel.empty:
        raise ValueError("None of the configured store-family series were found")
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    indexed_panel = panel.set_index("date").sort_index()
    origins = make_fold_origins(pd.Timestamp(indexed_panel.index.max()))
    folds = backtesting.generate_rolling_folds(
        indexed_panel,
        forecast_origins=origins,
        horizon=FORECAST_HORIZON,
    )
    if len(folds) != FOLD_COUNT:
        raise ValueError(
            f"Expected {FOLD_COUNT} complete folds, generated {len(folds)}"
        )

    result_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for fold in folds:
        validation = validate_fold_series_coverage(fold)
        for store_nbr, family in SERIES_CONFIG:
            result_rows.append(
                score_series_model(
                    fold,
                    validation,
                    store_nbr,
                    family,
                    "Naive",
                    models.NaiveModel(),
                )
            )
            result_rows.append(
                score_series_model(
                    fold,
                    validation,
                    store_nbr,
                    family,
                    "Seasonal Naive (7-day)",
                    models.SeasonalNaiveModel(seasonal_period=7),
                )
            )

    results = pd.DataFrame(result_rows)
    print("Per-series, per-fold baseline metrics")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    summary = results.groupby("model", as_index=False).agg(
        mean_rmsle=("rmsle", "mean"),
        median_rmsle=("rmsle", "median"),
        mean_mae=("mae", "mean"),
        mean_rmse=("rmse", "mean"),
    )
    print("\nAggregate metrics by model")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    for model_name, result_name, parameters in [
        ("Naive", "Naive", {"strategy": "latest observed value"}),
        ("Seasonal Naive", "Seasonal Naive (7-day)", {"seasonal_lag": 7}),
    ]:
        outcome = tracking.log_evaluation_run(
            model_name=model_name,
            model=None,
            model_parameters=parameters,
            results=results.loc[results["model"].eq(result_name)],
            forecast_horizon=FORECAST_HORIZON,
            number_of_series=len(SERIES_CONFIG),
            number_of_folds=FOLD_COUNT,
            forecast_origins=[
                fold.forecast_origin.strftime("%Y-%m-%d") for fold in folds
            ],
            runtime_seconds=time.perf_counter() - started,
            project_root=PROJECT_ROOT,
            tags={"model_family": "classical"},
            run_name=f"{model_name}-rolling-origin",
        )
        print(
            f"MLflow {model_name}: {outcome.status} {outcome.run_id or outcome.message}"
        )


if __name__ == "__main__":
    main()
