# PROJECT_CONTEXT.md

## 1. Project

Repository:

`https://github.com/anuragpandey1292/Time_Series_Forecasting_Complete_ML_ops`

Local path:

`C:\Users\Anuragh\Documents\Time_Series_Forecasting_Complete_ML_ops`

Environment:

- Windows
- VS Code / PowerShell
- Python 3.11
- `.venv`
- Git/GitHub
- planned MLflow, FastAPI, Docker, GCP deployment

Goal: build a complete production-style time-series forecasting + MLOps project demonstrating senior Data Scientist and MLOps skills.

## 2. Dataset

Favorita Grocery Sales.

Raw files:

```text
data/raw/
├── train.csv
├── test.csv
├── stores.csv
├── oil.csv
├── holidays_events.csv
└── transactions.csv
```

Key facts:

- train: 3,000,888 rows
- test: 28,512 rows
- train dates: 2013-01-01 to 2017-08-15
- test dates: 2017-08-16 to 2017-08-31
- 54 stores
- 33 families
- 1,782 store-family series
- train sales mean ≈ 357.8
- median = 11
- min = 0
- max ≈ 124,717
- no duplicate date/store/family rows
- no train/test date overlap
- `onpromotion` is present in future test data

## 3. Forecast contract

File:

`configs/forecasting_spec_v1.md`

Core:

- grain: `store_nbr × family × date`
- horizon: 16 days
- production origin: 2017-08-15
- production dates: 2017-08-16 through 2017-08-31
- use information available at/before origin unless a future feature is explicitly known in advance
- future actual sales are never model inputs
- rolling-origin backtesting is required

## 4. Feature availability

| Feature | Status |
|---|---|
| Date/calendar | available |
| Store metadata | available |
| Family | available |
| Historical sales | available through origin |
| `onpromotion` | available in future test |
| Holidays | known in advance |
| Oil | covers train/test and used in SARIMAX |
| Payday | deterministic |
| Future transactions | not confirmed; do not assume |
| Future actual sales | never allowed |

## 5. EDA findings

Notebook:

`notebooks/01_time_series_eda.ipynb`

Findings:

- upward trend from 2013 to 2017
- changing variance
- strong weekly seasonality
- substantial store heterogeneity
- `GROCERY I` is the dominant family
- sales are highly right-skewed
- promotions correlate with higher sales but this is not causal
- oil, holidays, transactions, calendar effects, payday, and historical events are candidate drivers

Approximate average sales by weekday:

- Monday ≈ 620K
- Tuesday ≈ 570K
- Wednesday ≈ 590K
- Thursday ≈ 510K
- Friday ≈ 580K
- Saturday ≈ 770K
- Sunday ≈ 820K

Store 44 is high-volume (~62M total in EDA), while store 52 is much smaller (~2.7M).

## 6. Holiday semantics

Locale:

- National: all stores
- Regional: matching state
- Local: matching city/locality

Transferred semantics:

- original `transferred=True` date is not automatically the closure date
- corresponding `Transfer` date is eligible as actual celebrated date

Current modeling-layer closure candidates:

- Holiday
- Additional
- Transfer
- Bridge

`Work Day` is not a closure and takes precedence.

Generic `Event` is not automatically a closure.

## 7. December 25 gap

Raw train has no rows for December 25 across store-family combinations in 2013-2016.

December 24 and 26 contain all 1,782 combinations.

Holiday metadata identifies December 25 as national `Navidad`.

Interpretation: likely dataset-wide closure, not observed zero sales.

Never modify raw data.

## 8. Architecture

```text
src/forecasting/
├── data/
│   ├── loader.py
│   ├── validation.py
│   └── imputation.py
├── backtesting/
│   └── rolling.py
├── models/
│   ├── base.py
│   ├── naive.py
│   ├── seasonal_naive.py
│   ├── arima.py
│   ├── sarima.py
│   └── sarimax.py
├── evaluation/
│   └── metrics.py
└── features/
    ├── calendar.py
    ├── lags.py
    ├── rolling.py
    ├── promotions.py
    └── pipeline.py
```

Tests mirror the components.

Scripts currently include profiling and evaluation scripts for baselines, ARIMA, SARIMA, and SARIMAX.

## 9. Backtesting

Rolling-origin, model-agnostic.

A fold includes:

- forecast origin
- train start/end
- validation start/end

Training ends at origin; validation follows immediately.

Current horizon: 16 days.

## 10. Baselines

Naive: latest observed sales.

Seasonal Naive: value from exactly 7 calendar days earlier.

Representative experiment:

Series:

- 44 × GROCERY I
- 1 × GROCERY I
- 10 × BEVERAGES
- 1 × PRODUCE
- 45 × BEVERAGES

Origins:

- 2017-06-28
- 2017-07-14
- 2017-07-30

Aggregate:

| Model | Mean RMSLE | Median RMSLE | Mean MAE | Mean RMSE |
|---|---:|---:|---:|---:|
| Naive | 0.4336 | 0.3569 | 1940.1267 | 2351.0481 |
| Seasonal Naive | 0.1697 | 0.1842 | 629.2666 | 832.8903 |

Seasonal Naive is strong but not universally best.

## 11. ARIMA

Specification:

`ARIMA(1,1,1)`

Experiment:

- 5 representative series
- 3 folds
- 15 fits

Aggregate:

- mean RMSLE: 0.3009
- median RMSLE: 0.2977
- mean MAE: 1221.6001
- mean RMSE: 1514.4389

ARIMA improved over Naive in this experiment but did not beat Seasonal Naive.

## 12. Feature engineering

Target lags use exact calendar dates:

- `lag_1` = t-1 day
- `lag_7` = t-7 days
- etc.

Missing source dates remain missing rather than causing farther-back lookup.

Rolling windows are calendar-based:

`[date-N days, date)`

Only information before the target date is used.

No automatic target synthesis or imputation occurs inside feature functions.

## 13. SARIMA

Specification:

`SARIMA(1,1,1)(1,0,1,7)`

Initial numerical failure: LU decomposition.

Final experiment:

- 14 successful fits
- 1 failed fit
- failed: Store 1 × PRODUCE, origin 2017-06-28
- failure: LU decomposition

Successful-only aggregate:

- mean RMSLE: 0.1889
- median RMSLE: 0.1778
- mean MAE: 881.6867
- mean RMSE: 1094.2058

Interpretation: better than ARIMA in the tested experiment, but numerical stability remains a concern.

## 14. Missing-data layer

Implementation:

`src/forecasting/data/imputation.py`

Tests:

`tests/test_imputation.py`

Design:

- raw data unchanged
- daily store-family grids can be completed in modeling layer
- holiday closures can be represented as zero
- audit columns include:
  - `sales_original`
  - `sales_imputation_method`
  - `is_imputed`

Important leakage discovery:

Backward-filling target values can leak future observations into earlier rolling-origin folds.

Example:

```text
Jan 2 = missing
Jan 3 = 20
```

Bfill would make Jan 2 = 20, which is future information if origin is Jan 2.

A modeling-safe API was therefore added:

`impute_training_sales_for_modeling`

It:

- accepts a `train_end`
- handles confirmed holiday closures as zero
- forward-fills ordinary missing training values only from earlier values
- leaves leading ordinary gaps NaN/unfilled
- does not backward-fill from future observations

## 15. SARIMAX

Specification:

`SARIMAX(1,1,1)(1,0,1,7)`

First exogenous feature set:

- `onpromotion`
- `oil_price`
- `holiday_indicator`
- `payday`

Not currently used:

- future transactions
- earthquake feature
- ML-style rolling target features

Initial issue: oil had a leading missing value on 2013-01-01.

Solution: model-specific handling excludes leading training rows with non-finite required exogenous values. Raw oil remains unchanged.

Initial statsmodels frequency issue occurred because missing calendar dates produced a date index without a usable frequency.

The SARIMAX implementation was updated to construct/align to an explicit daily DatetimeIndex. This resolved:

`ValueError: No supported index is available.`

## 16. Latest SARIMAX result

All 15 folds completed.

Per-fold results:

| Store | Family | Origin | RMSLE | MAE | RMSE |
|---:|---|---|---:|---:|---:|
| 44 | GROCERY I | 2017-06-28 | 0.1494 | 1353.3135 | 1692.3942 |
| 1 | GROCERY I | 2017-06-28 | 0.0776 | 145.0918 | 178.9324 |
| 10 | BEVERAGES | 2017-06-28 | 0.1973 | 191.7751 | 239.5250 |
| 1 | PRODUCE | 2017-06-28 | 0.0928 | 184.8693 | 224.9920 |
| 45 | BEVERAGES | 2017-06-28 | 0.1787 | 1536.1793 | 1810.8268 |
| 44 | GROCERY I | 2017-07-14 | 0.1226 | 1157.2509 | 1311.2002 |
| 1 | GROCERY I | 2017-07-14 | 0.1561 | 336.5429 | 362.2998 |
| 10 | BEVERAGES | 2017-07-14 | 0.1476 | 148.5303 | 185.7647 |
| 1 | PRODUCE | 2017-07-14 | 0.0764 | 146.5883 | 186.1994 |
| 45 | BEVERAGES | 2017-07-14 | 0.1133 | 1019.2782 | 1251.8809 |
| 44 | GROCERY I | 2017-07-30 | 0.2288 | 2122.7816 | 2563.8063 |
| 1 | GROCERY I | 2017-07-30 | **4.4293** | **175987.1880** | **176066.6091** |
| 10 | BEVERAGES | 2017-07-30 | 0.2296 | 263.6107 | 318.8540 |
| 1 | PRODUCE | 2017-07-30 | 0.1784 | 238.8971 | 321.6682 |
| 45 | BEVERAGES | 2017-07-30 | 0.1874 | 1327.0062 | 1779.7576 |

Aggregate:

```text
mean RMSLE   0.4377
median RMSLE 0.1561
mean MAE     12410.5935
mean RMSE    12566.3140
```

Warnings included repeated:

`ConvergenceWarning: Maximum Likelihood optimization failed to converge.`

and:

`EstimationWarning: Non-invertible starting MA parameters found. Using zeros as starting parameters.`

### Current conclusion

SARIMAX is not production-ready yet.

Median performance is strong, but the catastrophic fold makes robustness questionable.

Do not remove the fold or tune the model before diagnosing it.

## 17. Immediate next step

Diagnostic-only investigation:

- Store 1 × GROCERY I
- forecast origin: 2017-07-30
- validation: 2017-07-31 through 2017-08-15

Inspect:

- actual sales
- predictions
- prediction errors
- prediction magnitude
- convergence flag/status
- `mle_retvals`
- fitted parameters
- validation exogenous variables
- missing/non-finite exogenous values
- discontinuities

Do not change SARIMAX yet.

## 18. Test status

At the latest completed milestone:

`107 tests passed`

Ruff and format checks were also passing then.

Always rerun after changes.

Preferred:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 19. Git milestones

Known commits:

- `e361978` — forecasting specification
- `5bbe866` — rolling-origin backtesting
- `0cdc357` — feature engineering

A later calendar-semantics correction was also committed/pushed; exact hash is not recorded here.

Current SARIMA/SARIMAX/imputation work has included uncommitted changes unless subsequently committed.

Do not use `git add .`.

## 20. User workflow preference

The user prefers:

- 🟢 YOU DO
- 🤖 CODEX DO
- 🔍 YOU REVIEW
- 🧪 YOU TEST
- 📦 YOU COMMIT

They want exact Codex prompts and explanations so they learn the system rather than simply receiving generated code.

Avoid giant prompts and giant refactors.

The user plans a second pass after project completion to create a detailed PDF and a reusable skill/reference document. Preserve decisions, experiments, failures, and rationale.

## 21. Long-term roadmap

1. Finish robust classical model evaluation
2. Feature engineering experiments
3. ML models
4. Clustering/segmentation experiments
5. Experimental Prophet/deep learning benchmarks
6. Model selection
7. MLflow tracking
8. training pipeline
9. registry/versioning
10. FastAPI
11. Docker
12. GCP
13. monitoring/drift/performance
14. CI/CD
15. final documentation
16. portfolio PDF
17. reusable skill/reference file
