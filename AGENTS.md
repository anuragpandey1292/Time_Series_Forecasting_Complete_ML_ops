# AGENTS.md

## Project purpose

This repository is a production-style time-series forecasting and MLOps project built to demonstrate senior Data Scientist + MLOps engineering skills.

Primary forecasting problem:

- Daily sales forecasting at `store_nbr × family × date`
- 1,782 store-family series
- Favorita Grocery Sales dataset
- Production horizon: 16 days
- Production forecast origin: 2017-08-15
- Production forecast dates: 2017-08-16 through 2017-08-31

The project is developed incrementally. The goal is not only predictive accuracy, but also sound forecasting methodology, leakage prevention, reproducibility, testing, model comparison, MLOps, deployment, and monitoring.

## Codex working rules

### 1. Work incrementally

Do not generate a giant end-to-end codebase unless explicitly requested.

For each task:

1. Explain the intended change briefly.
2. Make the smallest coherent implementation.
3. Add/update focused tests.
4. Run relevant tests.
5. Run the full test suite when appropriate.
6. Run Ruff/format checks when appropriate.
7. Report what changed, what was tested, and remaining risks.

Prefer small, reviewable milestones.

### 2. Preserve architecture

Do not silently redesign the architecture.

If a request requires a substantial architectural change:

- explain why,
- identify the smallest safe change,
- ask for approval when appropriate.

Do not solve a local model failure by adding unrelated hacks to other layers.

### 3. Never modify raw data

`data/raw/` is source data.

Never overwrite raw CSVs, silently alter holiday definitions, or permanently impute raw observations.

Cleaning, completion, imputation, and transformations belong in the modeling/data-processing layer.

### 4. Leakage prevention is mandatory

For a forecast made at origin `T`:

- historical target values through `T` may be used;
- future target values after `T` must never be used;
- future features may be used only when genuinely known at forecast time;
- validation targets are for evaluation only.

Pay special attention to lags, rolling windows, imputation, recursive forecasting, exogenous variables, holidays, promotions, transactions, and oil.

### 5. Time-series feature semantics

Target lags are calendar-date based.

For target date `t`:

- `lag_1` = value at `t - 1 calendar day`
- `lag_7` = value at `t - 7 calendar days`

If the exact source date is absent, do not search farther backward.

Rolling windows are calendar based and use only information available before the target date.

### 6. Missing-data rules

Known closures can be encoded as zero in the modeling layer when supported by holiday semantics.

Current centralized layer:

`src/forecasting/data/imputation.py`

Important distinction:

- generic data preparation may use forward/backward filling only where explicitly intended;
- modeling-safe target preparation must not backward-fill from future observations.

Use the modeling-safe target API for rolling-origin model training.

Current holiday closure semantics:

- National applies to all stores
- Regional applies to matching state
- Local applies to matching city/locality
- transferred source dates are not the actual celebrated closure
- corresponding `Transfer` dates are eligible
- `Holiday`, `Additional`, `Transfer`, and `Bridge` can represent closures
- `Work Day` is not a closure and takes precedence
- generic `Event` is not automatically a closure

Do not invent additional closure rules without evidence.

### 7. Forecast-time availability

Current verified assumptions:

- date/calendar: available
- store metadata: available
- family: available
- historical sales: available through origin
- `onpromotion`: available in future test data
- holiday dates: known in advance
- oil: dataset covers train/test period and is currently used in SARIMAX
- payday: deterministic
- future transactions: not confirmed and must not be assumed
- future actual sales: never available to inference

If future availability is uncertain, do not use the feature in production without documenting and resolving the contract.

### 8. Model architecture

Classical forecasting models remain separate from ML feature-matrix pipelines.

Current models:

- Naive
- Seasonal Naive
- ARIMA
- SARIMA
- SARIMAX

Current fixed experimental specifications:

- ARIMA `(1,1,1)`
- SARIMA `(1,1,1)(1,0,1,7)`
- SARIMAX `(1,1,1)(1,0,1,7)`

Do not tune or change these merely to improve one fold without an explicit experiment.

### 9. Backtesting

Use rolling-origin evaluation.

Each fold has:

- forecast origin
- train start/end
- validation start/end

Training ends at the origin and validation follows immediately.

Current horizon: 16 days.

Never use validation targets as model inputs.

Never delete poor folds because they hurt aggregate metrics.

### 10. Metrics

Current metrics:

- RMSLE
- MAE
- RMSE

RMSLE compresses scale through a logarithmic transformation and emphasizes relative/log-scale differences. It does not inherently penalize underprediction more than overprediction.

### 11. Do not declare universal winners

Performance is heterogeneous across store-family series.

Current experiments show:

- Seasonal Naive is a strong baseline.
- ARIMA improved over Naive in the tested experiment.
- SARIMA improved over ARIMA but has had numerical instability.
- SARIMAX can be competitive but currently has a catastrophic fold and convergence warnings.

Treat results as experiment results, not universal conclusions.

Clustering or model-per-series strategies are hypotheses to test, not assumptions.

### 12. Current SARIMAX status

Latest SARIMAX experiment:

- 15/15 fits successful
- 0 failed fits
- multiple convergence warnings
- one catastrophic fold:

`Store 1 × GROCERY I, origin 2017-07-30`

Metrics:

- RMSLE: 4.4293
- MAE: 175987.1880
- RMSE: 176066.6091

Overall:

- mean RMSLE: 0.4377
- median RMSLE: 0.1561
- mean MAE: 12410.5935
- mean RMSE: 12566.3140

Do not remove the bad fold, suppress warnings, change the model specification, or alter aggregation before diagnosing it.

Next step is diagnostic-only investigation of that exact fold.

### 13. Testing

Always use the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Focused example:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_sarimax.py
```

Ruff:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

Do not rely on `python -m pytest` because Windows may resolve to Anaconda/system Python.

### 14. Git discipline

Do not commit unless explicitly asked.

Never use `git add .`.

Stage explicit files only.

Before committing:

1. inspect status,
2. inspect intended files,
3. run tests,
4. run Ruff,
5. stage intended files,
6. commit clearly,
7. push only when requested.

### 15. Diagnose before patching

For failures:

1. reproduce;
2. isolate smallest failing case;
3. inspect inputs;
4. inspect model state/parameters;
5. determine root cause;
6. propose smallest safe fix;
7. add regression test;
8. rerun focused tests;
9. rerun full suite.

For numerical failures, retain convergence and parameter evidence.

### 16. Learning objective

The user is learning senior-level forecasting and MLOps.

For substantial changes, explain:

- why the change is needed,
- the design principle,
- what can go wrong,
- what the user should review.

Do not hide important reasoning behind “done”.

### 17. Research/reference code

The earlier Kaggle-style script is research/reference material, not production code.

Use it for ideas, but review leakage, availability, missing-data semantics, testability, modularity, and reproducibility before adopting anything.

### 18. Scope

Broad roadmap:

1. data validation/profiling
2. forecasting specification
3. rolling backtesting
4. baselines
5. metrics
6. classical statistical models
7. exogenous modeling
8. feature engineering
9. model comparison
10. MLflow
11. training pipeline
12. registry/versioning
13. FastAPI
14. Docker
15. GCP
16. monitoring
17. CI/CD
18. final documentation/portfolio packaging

Do not jump ahead unless explicitly requested.

## Useful commands

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
```

Read both `AGENTS.md` and `PROJECT_CONTEXT.md` before modifying the project.
