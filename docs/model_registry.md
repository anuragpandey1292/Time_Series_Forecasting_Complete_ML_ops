# MLflow Model Registry

Experiment tracking records evaluation runs, parameters, metrics, and score
artifacts. The Model Registry stores deployable model artifacts as named
versions linked back to their source run.

The unified pipeline registers only the final LightGBM or CatBoost fit trained
with data available through the production origin, `2017-08-15`. Backtesting
models remain temporary and are never registered.

Register a LightGBM candidate:

```powershell
.\.venv\Scripts\python.exe scripts\train.py `
  --model lightgbm `
  --register-model true `
  --registered-model-name favorita-forecast-model `
  --model-alias candidate
```

Each successful registration creates a new model version. The optional
`candidate` alias points to the newly registered version. The pipeline never
assigns `champion`; promotion is an explicit later decision.

Inspect the local registry UI:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui `
  --backend-store-uri sqlite:///mlflow.db `
  --port 5000
```

The future FastAPI service should load the explicitly approved `champion`
alias, for example with an MLflow model URI like
`models:/favorita-forecast-model@champion`. It should not load `candidate`
automatically.
