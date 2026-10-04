"""Feature-level FastAPI inference service for the candidate MLflow model."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Any

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_FEATURES = [
    "store_nbr",
    "family",
    "onpromotion",
    "city",
    "state",
    "type",
    "cluster",
    "oil_price",
    "payday",
    "holiday_indicator",
    "year",
    "month",
    "quarter",
    "week",
    "day",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",
    "is_year_start",
    "is_year_end",
    "dow_sin",
    "dow_cos",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "lag_56",
    "lag_364",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "rolling_mean_28",
    "rolling_std_28",
    "rolling_mean_56",
    "rolling_std_56",
    "promotion_today",
    "promotion_lag_1",
    "promotion_rolling_7",
    "promotion_rolling_14",
]
CATEGORICAL_FEATURES = {"store_nbr", "family", "city", "state", "type", "cluster"}


class FeatureRow(BaseModel):
    """One fully prepared feature row accepted by the registered Booster."""

    model_config = ConfigDict(extra="forbid")
    store_nbr: int
    family: str
    onpromotion: float
    city: str
    state: str
    type: str
    cluster: int
    oil_price: float
    payday: int
    holiday_indicator: int
    year: int
    month: int
    quarter: int
    week: int
    day: int
    day_of_week: int
    day_of_year: int
    is_weekend: int
    is_month_start: int
    is_month_end: int
    is_quarter_start: int
    is_quarter_end: int
    is_year_start: int
    is_year_end: int
    dow_sin: float
    dow_cos: float
    lag_1: float | None = None
    lag_7: float | None = None
    lag_14: float | None = None
    lag_28: float | None = None
    lag_56: float | None = None
    lag_364: float | None = None
    rolling_mean_7: float | None = None
    rolling_std_7: float | None = None
    rolling_mean_14: float | None = None
    rolling_std_14: float | None = None
    rolling_mean_28: float | None = None
    rolling_std_28: float | None = None
    rolling_mean_56: float | None = None
    rolling_std_56: float | None = None
    promotion_today: float
    promotion_lag_1: float | None = None
    promotion_rolling_7: float | None = None
    promotion_rolling_14: float | None = None


class PredictionRequest(BaseModel):
    """Batch of prepared rows for feature-level prediction."""

    rows: list[FeatureRow] = Field(min_length=1, max_length=1000)


class PredictionResponse(BaseModel):
    """Numeric predictions aligned with the input row order."""

    predictions: list[float]


MODEL_NAME = os.getenv("FORECASTING_MODEL_NAME", "favorita-forecast-model")
MODEL_ALIAS = os.getenv("FORECASTING_MODEL_ALIAS", "candidate")
MODEL_URI = os.getenv(
    "FORECASTING_MODEL_URI",
    f"models:/{MODEL_NAME}@{MODEL_ALIAS}",
)
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
_model: Any | None = None
_model_error: str | None = None


def load_registered_model() -> Any:
    """Load the registered LightGBM native flavor once at application startup."""
    mlflow.set_tracking_uri(TRACKING_URI)
    return mlflow.lightgbm.load_model(MODEL_URI)


@asynccontextmanager
async def lifespan(application: FastAPI):
    """Load the candidate model once and retain load failures for health checks."""
    del application
    global _model, _model_error
    try:
        _model = load_registered_model()
        _model_error = None
    except Exception as exc:  # API must start so /health can explain the failure.
        _model = None
        _model_error = f"{type(exc).__name__}: {exc}"
    yield


app = FastAPI(title="Favorita Forecasting API", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, object]:
    """Report service and candidate-model loading status."""
    return {
        "status": "ok" if _model is not None else "error",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "model_loaded": _model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Predict from rows already prepared with the registered model features."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Forecasting model is not loaded")
    frame = pd.DataFrame([row.model_dump() for row in request.rows])
    missing = sorted(set(MODEL_FEATURES) - set(frame.columns))
    if missing:
        raise HTTPException(
            status_code=422, detail=f"Missing required features: {missing}"
        )
    frame = frame.loc[:, MODEL_FEATURES]
    category_levels = getattr(_model, "pandas_categorical", None)
    if not category_levels or len(category_levels) != len(CATEGORICAL_FEATURES):
        raise HTTPException(
            status_code=500, detail="Model categorical schema is unavailable"
        )
    categorical_order = ["store_nbr", "family", "city", "state", "type", "cluster"]
    for column, levels in zip(categorical_order, category_levels, strict=True):
        values = frame[column].astype(str)
        unknown = sorted(set(values) - set(levels))
        if unknown:
            raise HTTPException(
                status_code=422,
                detail=f"Unknown values for categorical feature {column}: {unknown}",
            )
        frame[column] = pd.Categorical(values, categories=levels)
    try:
        values = _model.predict(frame)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise HTTPException(
            status_code=422, detail=f"Incompatible feature values: {type(exc).__name__}"
        ) from exc
    if not pd.Series(values).map(pd.api.types.is_number).all():
        raise HTTPException(
            status_code=500, detail="Model returned non-numeric predictions"
        )
    return PredictionResponse(predictions=[max(0.0, float(value)) for value in values])
