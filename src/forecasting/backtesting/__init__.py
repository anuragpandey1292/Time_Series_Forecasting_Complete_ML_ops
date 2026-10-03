"""Model-agnostic time-series backtesting utilities."""

from forecasting.backtesting.rolling import BacktestFold, generate_rolling_folds

__all__ = ["BacktestFold", "generate_rolling_folds"]
