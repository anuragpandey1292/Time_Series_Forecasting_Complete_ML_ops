"""Small numeric metrics for comparing actual and predicted values."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _validated_arrays(
    actual: ArrayLike, predicted: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Convert inputs to numeric vectors and validate shape and length."""
    try:
        actual_values = np.asarray(actual, dtype=np.float64)
        predicted_values = np.asarray(predicted, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("actual and predicted must contain numeric values") from exc

    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("actual and predicted must be one-dimensional")
    if len(actual_values) != len(predicted_values):
        raise ValueError("actual and predicted must have equal lengths")
    if len(actual_values) == 0:
        raise ValueError("actual and predicted must not be empty")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("actual and predicted must contain only finite values")
    return actual_values, predicted_values


def rmsle(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return root mean squared logarithmic error.

    RMSLE is the root mean squared difference between ``log1p(actual)`` and
    ``log1p(predicted)``. Both inputs must be nonnegative. It is not inherently
    asymmetric: under-prediction is not automatically penalized more than
    over-prediction.
    """
    actual_values, predicted_values = _validated_arrays(actual, predicted)
    if np.any(actual_values < 0):
        raise ValueError("actual values must be nonnegative for RMSLE")
    if np.any(predicted_values < 0):
        raise ValueError("predicted values must be nonnegative for RMSLE")
    log_errors = np.log1p(predicted_values) - np.log1p(actual_values)
    return float(np.sqrt(np.mean(np.square(log_errors))))


def mae(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return mean absolute error between actual and predicted values."""
    actual_values, predicted_values = _validated_arrays(actual, predicted)
    return float(np.mean(np.abs(actual_values - predicted_values)))


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Return root mean squared error between actual and predicted values."""
    actual_values, predicted_values = _validated_arrays(actual, predicted)
    return float(np.sqrt(np.mean(np.square(actual_values - predicted_values))))
