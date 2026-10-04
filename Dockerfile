# Production-oriented Dockerfile for Favorita Forecasting FastAPI Service
FROM python:3.11-slim

WORKDIR /app

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Copy project metadata and source code needed for runtime
COPY pyproject.toml .
COPY src/ src/

# Install system dependencies needed for LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install runtime dependencies and project in editable/standard mode
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Default environment variables for model registry access
ENV FORECASTING_MODEL_NAME="favorita-forecast-model" \
    FORECASTING_MODEL_ALIAS="candidate" \
    MLFLOW_TRACKING_URI="sqlite:////app/mlflow.db"

EXPOSE 8000

# Run Uvicorn production server
CMD ["uvicorn", "forecasting.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
