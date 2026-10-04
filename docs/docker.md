# Docker Deployment Guide for Favorita Forecasting Service

## Overview

This document describes how to build, run, and configure the Docker container for the Favorita Forecasting FastAPI service.

## Environment Variables

The container relies on the following environment variables for model loading and MLflow tracking:

| Variable | Default Value | Description |
| --- | --- | --- |
| `FORECASTING_MODEL_NAME` | `favorita-forecast-model` | Registered model name in MLflow Model Registry |
| `FORECASTING_MODEL_ALIAS` | `candidate` | Model alias/tag to load (`models:/favorita-forecast-model@candidate`) |
| `MLFLOW_TRACKING_URI` | `sqlite:////app/mlflow.db` | URI pointing to the MLflow tracking store |

## How Model Loading Works

At application startup (`lifespan` context manager in `forecasting.api.app`):
1. FastAPI initializes and sets `mlflow.set_tracking_uri(TRACKING_URI)`.
2. It fetches the model artifact using `mlflow.lightgbm.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")`.
3. If successful, the model reference is saved globally for serving `/predict` requests.
4. If loading fails, `/health` reports `model_loaded: false` and `/predict` responds with `503 Service Unavailable`.

## Local MLflow SQLite & Artifact Portability

In local development, MLflow uses a local SQLite database (`mlflow.db`) and a local artifact store.

### Why the Absolute URI Failed in Docker
Previously, when the model was trained on the Windows host, MLflow captured the `artifact_location` as an absolute Windows URI (e.g., `file:///C:/Users/.../mlartifacts/...`). When the Docker container (a Linux environment) tried to load the model from the mounted `mlflow.db`, it attempted to read that `file:///C:/...` path, which does not exist in the Linux container filesystem, causing a model loading failure.

### The Portable Solution
To fix this without hardcoding paths or manually editing the SQLite database:
1. The `EXPERIMENT_NAME` was updated to `favorita-forecasting-portable` and the `artifact_location` to the relative path `mlartifacts/favorita-forecasting-portable` in the tracking setup.
2. The model training script was run **inside** a Docker container using a volume mount.
3. This allowed MLflow to record a portable Linux absolute path (`file:///app/mlartifacts/...`) into the SQLite database.
4. Because the `mlflow.db` and `mlartifacts` directory are mounted into the container at `/app`, the path resolves correctly inside the container, while the host Windows environment can also resolve it natively by interpreting the mounted paths, provided the host is aware of the context or we evaluate it within the container.

### Mounting Local MLflow for Local Container Testing

For local Docker testing, mount the host MLflow storage into the container volume so the `/app` paths align:

```powershell
docker run -d --name favorita-api -p 8000:8000 `
  -v "${PWD}/mlflow.db:/app/mlflow.db" `
  -v "${PWD}/mlartifacts:/app/mlartifacts" `
  -e MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db" `
  favorita-forecasting-api:latest
```

## Docker Build & Run Commands

### 1. Build Image

```powershell
docker build -t favorita-forecasting-api:latest .
```

### 2. Run Container (Local MLflow Mount)

```powershell
docker run -d --name favorita-api -p 8000:8000 `
  -v "${PWD}/mlflow.db:/app/mlflow.db" `
  -v "${PWD}/mlruns:/app/mlruns" `
  -e MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db" `
  favorita-forecasting-api:latest
```

## Health Check & Inference Flow

### GET /health

Request:
```powershell
curl http://localhost:8000/health
```

Expected Response:
```json
{
  "status": "ok",
  "model_name": "favorita-forecast-model",
  "model_alias": "candidate",
  "model_loaded": true
}
```

### POST /predict

Request:
```powershell
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "rows": [
      {
        "store_nbr": 1,
        "family": "GROCERY I",
        "onpromotion": 1.0,
        "city": "Quito",
        "state": "Pichincha",
        "type": "A",
        "cluster": 13,
        "oil_price": 45.0,
        "payday": 0,
        "holiday_indicator": 0,
        "year": 2017,
        "month": 8,
        "quarter": 3,
        "week": 33,
        "day": 16,
        "day_of_week": 2,
        "day_of_year": 228,
        "is_weekend": 0,
        "is_month_start": 0,
        "is_month_end": 0,
        "is_quarter_start": 0,
        "is_quarter_end": 0,
        "is_year_start": 0,
        "is_year_end": 0,
        "dow_sin": 0.9749,
        "dow_cos": -0.2225,
        "lag_1": 120.0,
        "lag_7": 115.0,
        "lag_14": 110.0,
        "lag_28": 105.0,
        "lag_56": 100.0,
        "lag_364": 95.0,
        "rolling_mean_7": 118.0,
        "rolling_std_7": 5.0,
        "rolling_mean_14": 114.0,
        "rolling_std_14": 6.0,
        "rolling_mean_28": 112.0,
        "rolling_std_28": 7.0,
        "rolling_mean_56": 108.0,
        "rolling_std_56": 8.0,
        "promotion_today": 1.0,
        "promotion_lag_1": 1.0,
        "promotion_rolling_7": 1.0,
        "promotion_rolling_14": 1.0
      }
    ]
  }'
```

Expected Response:
```json
{
  "predictions": [
    124.5
  ]
}
```

> [!NOTE]
> All predictions returned by `/predict` are guaranteed to be non-negative ($\ge 0.0$).
