# Time Series Forecasting - Complete ML Ops

## Project Overview

This repository will contain a production-style, end-to-end time-series forecasting system based on the Kaggle Favorita Grocery Sales dataset. The goal is to develop a reproducible forecasting workflow that progresses from raw data through deployment, monitoring, and automated retraining.

The project is currently in the **initialization stage**. The architecture, data contracts, and forecasting objective will be refined as the dataset is explored.

## Business Problem

The initial understanding is to forecast daily sales at the store x product-family level for a 15-day horizon. This definition, including the target, aggregation level, business constraints, and evaluation criteria, will be confirmed during data exploration.

## Dataset

The project uses the [Favorita Grocery Sales Forecasting dataset](https://www.kaggle.com/competitions/favorita-grocery-sales-forecasting) from Kaggle. It contains historical sales and related retail data intended for grocery sales forecasting.

## Planned Architecture

```text
Data Sources
  -> Data Validation
  -> Feature Engineering
  -> Time-Series Backtesting
  -> Model Training
  -> MLflow
  -> Model Registry
  -> FastAPI
  -> Docker
  -> GitHub Actions
  -> GCP
  -> Monitoring
  -> Retraining
```

This flow will support versioned data, repeatable experiments, validated model promotion, deployment, and ongoing model operations.

## Technology Stack

- Python for data processing, modeling, and services
- DVC for data versioning
- MLflow for experiment tracking and model registry
- FastAPI for model serving
- Docker for containerization
- pytest, Ruff, and mypy for testing and code quality
- GitHub Actions for CI/CD
- GCP for deployment and managed infrastructure

## Project Structure

The repository will evolve toward the following structure:

```text
data/                # DVC-managed datasets
notebooks/           # Exploratory analysis
src/                 # Application and forecasting code
tests/               # Automated tests
models/              # Model definitions and related code
configs/             # Configuration files
docker/              # Container assets
.github/workflows/   # CI/CD workflows
```

## Development Roadmap

1. Define the business problem and explore the data.
2. Build data ingestion, versioning, and validation workflows.
3. Perform exploratory time-series analysis and feature engineering.
4. Establish forecasting baselines and evaluation metrics.
5. Train statistical and machine-learning forecasting models.
6. Implement walk-forward backtesting and model evaluation.
7. Add MLflow tracking and model registry workflows.
8. Serve approved models through FastAPI and Docker.
9. Add automated tests and GitHub Actions CI/CD.
10. Deploy to GCP and introduce monitoring and automated retraining.

## Current Project Status

**Initialization stage.** Project documentation and repository conventions are being established. No data pipeline, model, deployment service, or performance result is implemented or reported yet.
