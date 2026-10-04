# Forecasting API

The service loads the registered LightGBM candidate model from:

```text
models:/favorita-forecast-model@candidate
```

The model is loaded once during application startup. The current registered
artifact is a feature-level LightGBM Booster, not a raw-data forecasting
pipeline. Therefore `/predict` accepts already prepared feature rows. It does
not synthesize dates, calculate recursive lags, or infer store metadata.

## Configuration

Defaults can be overridden with environment variables:

```text
FORECASTING_MODEL_NAME=favorita-forecast-model
FORECASTING_MODEL_ALIAS=candidate
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
```

Start the local service from the repository root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn forecasting.api.app:app --app-dir src --host 127.0.0.1 --port 8000
```

Check health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Example prediction request shape:

```json
{
  "rows": [
    {
      "store_nbr": 1,
      "family": "GROCERY I",
      "onpromotion": 0,
      "city": "Quito",
      "state": "Pichincha",
      "type": "D",
      "cluster": 13,
      "oil_price": 53.2,
      "payday": 0,
      "holiday_indicator": 0,
      "year": 2017,
      "month": 8,
      "quarter": 3,
      "week": 31,
      "day": 1,
      "day_of_week": 1,
      "day_of_year": 213,
      "is_weekend": 0,
      "is_month_start": 1,
      "is_month_end": 0,
      "is_quarter_start": 0,
      "is_quarter_end": 0,
      "is_year_start": 0,
      "is_year_end": 0,
      "dow_sin": 0.78,
      "dow_cos": 0.62,
      "lag_1": 1200,
      "lag_7": 1100,
      "lag_14": 1150,
      "lag_28": 1000,
      "lag_56": 980,
      "lag_364": 900,
      "rolling_mean_7": 1100,
      "rolling_std_7": 100,
      "rolling_mean_14": 1080,
      "rolling_std_14": 120,
      "rolling_mean_28": 1050,
      "rolling_std_28": 130,
      "rolling_mean_56": 1020,
      "rolling_std_56": 140,
      "promotion_today": 0,
      "promotion_lag_1": 0,
      "promotion_rolling_7": 0,
      "promotion_rolling_14": 0
    }
  ]
}
```

The response is aligned to input order:

```json
{"predictions": [1234.5]}
```

This milestone does not promote `champion`. A future service should load an
explicitly approved champion alias, such as
`models:/favorita-forecast-model@champion`, after a separate promotion step.
