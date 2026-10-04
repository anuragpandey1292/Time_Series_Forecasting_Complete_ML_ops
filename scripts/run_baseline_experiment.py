"""Compare naive baselines on one historical Favorita series."""

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

STORE_NBR = 44
FAMILY = "GROCERY I"
FORECAST_HORIZON = 16


def evaluate_model(
    model_name: str, model: models.ForecastModel, fold: backtesting.BacktestFold
) -> dict[str, object]:
    """Fit on fold training rows and score predictions against held-out sales."""
    train_data = fold.train_data.reset_index()
    future_data = fold.validation_features.reset_index()

    model.fit(train_data)
    predictions = model.predict(future_data)

    actuals = (
        fold.validation_target.rename("actual")
        .rename_axis("date")
        .reset_index()
    )
    actuals["store_nbr"] = STORE_NBR
    actuals["family"] = FAMILY
    scored = predictions.merge(
        actuals,
        on=["date", "store_nbr", "family"],
        how="inner",
        validate="one_to_one",
    )
    if len(scored) != FORECAST_HORIZON:
        raise ValueError(
            "Prediction and validation rows did not align for the full horizon"
        )

    return {
        "model": model_name,
        "forecast_origin": fold.forecast_origin.date(),
        "validation_start": fold.validation_start.date(),
        "validation_end": fold.validation_end.date(),
        "RMSLE": metrics.rmsle(scored["actual"], scored["prediction"]),
        "MAE": metrics.mae(scored["actual"], scored["prediction"]),
        "RMSE": metrics.rmse(scored["actual"], scored["prediction"]),
    }


def main() -> None:
    """Run and print a two-model baseline comparison for the chosen series."""
    train = data_loader.load_train()
    series = train.loc[
        (train["store_nbr"] == STORE_NBR) & (train["family"] == FAMILY),
        ["date", "store_nbr", "family", "sales"],
    ].copy()
    if series.empty:
        raise ValueError(
            f"No training observations found for store {STORE_NBR}, {FAMILY!r}"
        )

    series["date"] = pd.to_datetime(series["date"], errors="raise")
    indexed_series = series.set_index("date").sort_index()
    forecast_origin = indexed_series.index.max() - pd.Timedelta(
        days=FORECAST_HORIZON
    )
    folds = backtesting.generate_rolling_folds(
        indexed_series,
        forecast_origins=[forecast_origin],
        horizon=FORECAST_HORIZON,
    )
    if len(folds) != 1:
        raise ValueError(
            "Could not create a complete historical validation fold for "
            f"store {STORE_NBR}, {FAMILY!r} at {forecast_origin.date()}"
        )
    fold = folds[0]

    results = [
        evaluate_model("Naive", models.NaiveModel(), fold),
        evaluate_model(
            "Seasonal Naive (7-day)",
            models.SeasonalNaiveModel(seasonal_period=7),
            fold,
        ),
    ]
    comparison = pd.DataFrame(results)
    print(f"Baseline experiment: store_nbr={STORE_NBR}, family={FAMILY}")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
