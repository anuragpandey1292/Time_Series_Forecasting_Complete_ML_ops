"""Tests for the forecasting evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from forecasting.evaluation import mae, rmse, rmsle


@pytest.mark.parametrize("metric", [mae, rmse, rmsle])
def test_perfect_predictions_return_zero(metric) -> None:
    """Perfect predictions, including zero targets, have zero error."""
    actual = pd.Series([0.0, 2.0, 5.0])
    predicted = pd.Series([0.0, 2.0, 5.0])

    assert metric(actual, predicted) == 0.0


def test_rmsle_handles_zero_actual_values() -> None:
    """Zero actual values are handled by log1p without special casing errors."""
    actual = np.array([0.0, 1.0])
    predicted = np.array([1.0, 1.0])

    assert rmsle(actual, predicted) == pytest.approx(np.log(2) / np.sqrt(2))


def test_rmsle_matches_hand_calculated_example() -> None:
    """RMSLE matches the root mean square of the two log differences."""
    actual = [0.0, 1.0, 2.0]
    predicted = [0.0, 2.0, 1.0]
    expected = np.sqrt((2 * np.log(1.5) ** 2) / 3)

    assert rmsle(actual, predicted) == pytest.approx(expected)


def test_mae_matches_hand_calculated_example() -> None:
    """MAE is the average absolute error for a small example."""
    assert mae([1, 2, 3], [2, 2, 5]) == pytest.approx(1.0)


def test_rmse_matches_hand_calculated_example() -> None:
    """RMSE is the square root of the average squared error."""
    assert rmse([1, 2, 3], [2, 2, 5]) == pytest.approx(np.sqrt(5 / 3))


@pytest.mark.parametrize("predicted", [[0.0, -0.1], [-1.0]])
def test_rmsle_rejects_negative_predictions(predicted: list[float]) -> None:
    """RMSLE rejects predictions outside the logarithm's valid domain."""
    actual = [0.0, 1.0] if len(predicted) == 2 else [0.0]

    with pytest.raises(ValueError, match="predicted values must be nonnegative"):
        rmsle(actual, predicted)


@pytest.mark.parametrize("metric", [mae, rmse, rmsle])
def test_metrics_reject_mismatched_lengths(metric) -> None:
    """Every metric rejects vectors of different lengths."""
    with pytest.raises(ValueError, match="equal lengths"):
        metric([1.0, 2.0], [1.0])


def test_metrics_do_not_modify_inputs() -> None:
    """Metric calculations leave the caller's arrays unchanged."""
    actual = np.array([0.0, 2.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0])
    actual_before = actual.copy()
    predicted_before = predicted.copy()

    mae(actual, predicted)
    rmse(actual, predicted)
    rmsle(actual, predicted)

    np.testing.assert_array_equal(actual, actual_before)
    np.testing.assert_array_equal(predicted, predicted_before)
