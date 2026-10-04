"""Global LightGBM forecasting model with recursive target features."""

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.models.global_ml import GlobalRecursiveRegressor


class LightGBMModel(GlobalRecursiveRegressor):
    """Fixed global LightGBM regressor for store-family daily forecasts."""

    def __init__(self) -> None:
        """Create the initial fixed regression configuration."""
        self._params = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": 42,
            "num_threads": 8,
            "verbosity": -1,
        }
        self._num_boost_round = 100
        super().__init__(None, categorical_mode="lightgbm")

    def _fit_estimator(self, matrix: pd.DataFrame, target: np.ndarray) -> None:
        """Fit LightGBM directly with categorical pandas columns."""
        dataset = lgb.Dataset(
            matrix,
            label=target,
            categorical_feature=self._categorical_columns,
            free_raw_data=False,
        )
        self._estimator = lgb.train(
            self._params,
            dataset,
            num_boost_round=self._num_boost_round,
        )
