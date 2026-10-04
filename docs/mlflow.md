# Local MLflow Tracking

The ML evaluation script logs one run for each global tree model in the
`favorita-forecasting` experiment. Metadata is stored in the ignored
`mlflow.db` SQLite database, and score-only artifacts are stored under the
ignored `mlartifacts/` directory. Raw Favorita rows are not logged.

Run the representative LightGBM and CatBoost evaluation from the repository
root:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_ml_models.py
```

Tracking is enabled by default. Set `FORECASTING_MLFLOW_ENABLED=false` to run
the evaluation without tracking.

Launch the local UI from the repository root:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

Open `http://127.0.0.1:5000` to inspect runs.
