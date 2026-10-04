"""Evaluate fixed-order SARIMA on representative Favorita series and folds."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
MODEL_NAME = "SARIMA(1,1,1)(1,0,1,7)"


def prepare_validation(fold: backtesting.BacktestFold) -> pd.DataFrame:
    """Pair held-out targets with keys for scoring, not model input."""
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
) -> dict[str, object]:
    """Fit one SARIMA model to a fold's training history and score it."""
    train_mask = (fold.train_data["store_nbr"] == store_nbr) & (
        fold.train_data["family"] == family
    )
    train_data = fold.train_data.loc[train_mask].reset_index()
    validation_mask = (validation["store_nbr"] == store_nbr) & (
        validation["family"] == family
    )
    series_validation = validation.loc[validation_mask]

    result = {
        "store_nbr": store_nbr,
        "family": family,
        "forecast_origin": fold.forecast_origin.date(),
        "validation_start": fold.validation_start.date(),
        "validation_end": fold.validation_end.date(),
        "model": MODEL_NAME,
    }

    model = models.SARIMAModel()
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

    predictions = model.predict(series_validation.drop(columns="actual"))
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
        **result,
        "rmsle": metrics.rmsle(scored["actual"], scored["prediction"]),
        "mae": metrics.mae(scored["actual"], scored["prediction"]),
        "rmse": metrics.rmse(scored["actual"], scored["prediction"]),
        "status": "success",
        "error": None,
    }


def evaluate_folds(
    folds: list[backtesting.BacktestFold],
) -> pd.DataFrame:
    """Evaluate every configured series and retain individual fit failures."""
    result_rows = []
    for fold in folds:
        validation = prepare_validation(fold)
        for store_nbr, family in SERIES_CONFIG:
            result_rows.append(score_series(fold, validation, store_nbr, family))
    return pd.DataFrame(result_rows)


def report_results(results: pd.DataFrame) -> None:
    """Print successful scores, failed fits, and success-only aggregates."""
    successful = results.loc[results["status"].eq("success")]
    failed = results.loc[results["status"].eq("fit_failed")]

    print("Successful per-series, per-fold SARIMA metrics")
    display_columns = [
        "store_nbr",
        "family",
        "forecast_origin",
        "validation_start",
        "validation_end",
        "model",
        "rmsle",
        "mae",
        "rmse",
    ]
    print(
        successful.loc[:, display_columns].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        )
    )

    print("\nFailed SARIMA fits")
    failed_columns = [
        "store_nbr",
        "family",
        "forecast_origin",
        "validation_start",
        "validation_end",
        "error",
    ]
    if failed.empty:
        print("None")
    else:
        print(failed.loc[:, failed_columns].to_string(index=False))

    summary = {
        "mean_rmsle": successful["rmsle"].mean(),
        "median_rmsle": successful["rmsle"].median(),
        "mean_mae": successful["mae"].mean(),
        "mean_rmse": successful["rmse"].mean(),
    }
    print("\nAggregate SARIMA summary (successful fits only)")
    print(f"Successful fits: {len(successful)}")
    print(f"Failed fits: {len(failed)}")
    for metric_name, value in summary.items():
        print(
            f"{metric_name}: {value:.4f}" if pd.notna(value) else f"{metric_name}: NaN"
        )


def main() -> None:
    """Evaluate the fixed SARIMA model for five series and three origins."""
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
    folds = backtesting.generate_rolling_folds(
        panel.set_index("date").sort_index(),
        forecast_origins=FORECAST_ORIGINS,
        horizon=FORECAST_HORIZON,
    )
    if len(folds) != len(FORECAST_ORIGINS):
        raise ValueError(
            f"Expected {len(FORECAST_ORIGINS)} complete folds, generated {len(folds)}"
        )

    report_results(evaluate_folds(folds))


if __name__ == "__main__":
    main()
