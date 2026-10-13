"""Optuna Bayesian hyperparameter tuning CLI for LightGBM on rolling-origin folds."""

from __future__ import annotations

import argparse
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

backtesting = import_module("forecasting.backtesting")
target_imputation = import_module("forecasting.data.imputation")
data_loader = import_module("forecasting.data.loader")
feature_known = import_module("forecasting.features.known")
models = import_module("forecasting.models")
tracking = import_module("forecasting.tracking")
tuning = import_module("forecasting.tuning")

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


def select_panel(raw_train: pd.DataFrame) -> pd.DataFrame:
    """Filter to representative series panel without altering raw rows."""
    mask = pd.Series(False, index=raw_train.index)
    for store_nbr, family in SERIES_CONFIG:
        mask |= raw_train["store_nbr"].eq(store_nbr) & raw_train["family"].eq(family)
    panel = raw_train.loc[mask].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="raise")
    return panel


def prepare_folds_data(
    panel: pd.DataFrame,
    stores: pd.DataFrame,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
) -> list[tuning.FoldData]:
    """Precompute training and validation feature frames once for all trials."""
    folds = backtesting.generate_rolling_folds(
        panel.set_index("date").sort_index(),
        forecast_origins=FORECAST_ORIGINS,
        horizon=FORECAST_HORIZON,
    )
    precomputed: list[tuning.FoldData] = []
    for fold in folds:
        print(
            f"Precomputing features for fold {fold.forecast_origin.date()}...",
            flush=True,
        )
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
        train_features = feature_known.add_known_features(
            imputed, stores, oil_prices, holidays
        )

        val_reset = fold.validation_features.reset_index()
        val_reset["actual"] = fold.validation_target.to_numpy()
        selected_val = val_reset.loc[
            val_reset.set_index(SERIES_COLUMNS).index.isin(SERIES_CONFIG)
        ].copy()
        future_features = feature_known.add_known_features(
            selected_val.drop(columns="actual"),
            stores,
            oil_prices,
            holidays,
        )
        actuals = selected_val.loc[:, [*KEY_COLUMNS, "actual"]]

        precomputed.append(
            tuning.FoldData(
                forecast_origin=fold.forecast_origin,
                train_features=train_features,
                future_features=future_features,
                actuals=actuals,
            )
        )
    return precomputed


def train_and_register_tuned_model(
    best_params: dict[str, Any],
    best_num_boost_round: int,
    panel: pd.DataFrame,
    stores: pd.DataFrame,
    oil_prices: pd.Series,
    holidays: pd.DataFrame,
    registered_model_name: str,
    alias: str | None,
    best_metrics: dict[str, float],
) -> None:
    """Retrain champion configuration through origin and register in MLflow."""
    import mlflow

    production_origin = pd.Timestamp("2017-08-15")
    rows = panel.loc[panel["date"].le(production_origin)].copy()
    imputed = target_imputation.impute_training_sales_for_modeling(
        rows.loc[:, [*KEY_COLUMNS, "sales", "onpromotion"]],
        stores,
        holidays,
        train_end=production_origin,
        complete_daily_grid=True,
    )
    training = feature_known.add_known_features(imputed, stores, oil_prices, holidays)
    model = models.LightGBMModel(
        params=best_params, num_boost_round=best_num_boost_round
    )
    model.fit(training)

    db_path = PROJECT_ROOT / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve().as_posix()}")
    mlflow.set_experiment("favorita-tuned-champion")

    with mlflow.start_run(run_name="lightgbm-tuned-champion") as run:
        all_params = {**best_params, "num_boost_round": best_num_boost_round}
        mlflow.log_params(all_params)
        mlflow.log_metrics(best_metrics)
        from mlflow import lightgbm as mlflow_lgb

        mlflow_lgb.log_model(model._estimator, artifact_path="model")
        reg_version = tracking.register_model(
            run_id=run.info.run_id,
            artifact_path="model",
            registered_model_name=registered_model_name,
            alias=alias,
            tags={
                "algorithm": "lightgbm-tuned",
                "mean_rmsle": str(best_metrics.get("mean_rmsle")),
                "mean_mae": str(best_metrics.get("mean_mae")),
                "mean_rmse": str(best_metrics.get("mean_rmse")),
                "training_origin": "2017-08-15",
                "tuned_by": "optuna",
            },
        )
        print(
            f"Successfully registered tuned model version {reg_version.version} "
            f"with alias '{alias}'"
        )


def parse_args() -> argparse.Namespace:
    """Parse CLI options for tuning study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--metric",
        default="mean_rmsle",
        choices=["mean_rmsle", "mean_mae", "mean_rmse"],
    )
    parser.add_argument("--mlflow-enabled", default="true", choices=["true", "false"])
    parser.add_argument("--register-best", default="false", choices=["true", "false"])
    parser.add_argument("--registered-model-name", default="favorita-forecast-model")
    parser.add_argument("--model-alias", default="tuned-candidate")
    return parser.parse_args()


def main() -> None:
    """Entry point for Bayesian hyperparameter tuning."""
    args = parse_args()
    print("=" * 70)
    print(f"Starting LightGBM Bayesian Hyperparameter Tuning ({args.n_trials} trials)")
    print("=" * 70)

    t0 = time.perf_counter()
    raw_train = data_loader.load_train()
    panel = select_panel(raw_train)
    stores = data_loader.load_stores()
    oil_prices = feature_known.prepare_oil_prices(data_loader.load_oil())
    holidays = data_loader.load_holidays_events().copy()
    holidays["date"] = pd.to_datetime(holidays["date"], errors="raise")

    folds_data = prepare_folds_data(panel, stores, oil_prices, holidays)
    precompute_time = time.perf_counter() - t0
    print(f"Precomputed 3 folds in {precompute_time:.1f}s. Running optimization...")

    tuner = tuning.LightGBMTuner(folds=folds_data, metric=args.metric, seed=42)
    result = tuner.optimize(
        n_trials=args.n_trials,
        study_name="favorita-lightgbm-tuning",
        mlflow_tracking=args.mlflow_enabled == "true",
        project_root=PROJECT_ROOT,
    )

    print("\n" + "=" * 70)
    print("Tuning Completed!")
    print(f"Best Trial: #{result.best_trial_number}")
    print(f"Best {args.metric}: {result.best_value:.4f}")
    print(f"Best num_boost_round: {result.best_num_boost_round}")
    print("Best Hyperparameters:")
    for param_name, param_val in sorted(result.best_params.items()):
        if param_name not in {"num_threads", "verbosity", "objective"}:
            print(f"  {param_name}: {param_val}")
    print("=" * 70)

    baseline_rmsle = 0.1369
    improvement = (baseline_rmsle - result.best_value) / baseline_rmsle * 100
    print(f"\nBaseline Untuned LightGBM Mean RMSLE: {baseline_rmsle:.4f}")
    print(f"Tuned LightGBM Mean RMSLE:            {result.best_value:.4f}")
    if improvement > 0:
        print(f"Relative Improvement:                 +{improvement:.2f}%")
    else:
        print(f"Relative Difference:                  {improvement:.2f}%")

    if args.register_best == "true":
        print("\nRetraining champion configuration through origin 2017-08-15...")
        best_eval = tuner.evaluate_configuration(
            result.best_params, result.best_num_boost_round
        )
        train_and_register_tuned_model(
            best_params=result.best_params,
            best_num_boost_round=result.best_num_boost_round,
            panel=panel,
            stores=stores,
            oil_prices=oil_prices,
            holidays=holidays,
            registered_model_name=args.registered_model_name,
            alias=args.model_alias,
            best_metrics=best_eval,
        )


if __name__ == "__main__":
    main()
