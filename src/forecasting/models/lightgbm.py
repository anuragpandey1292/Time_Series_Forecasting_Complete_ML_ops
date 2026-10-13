from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from forecasting.models.global_ml import GlobalRecursiveRegressor


class LightGBMModel(GlobalRecursiveRegressor):
    """Global LightGBM regressor with customizable hyperparameters."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 100,
    ) -> None:
        """Create the regression configuration with optional overrides."""
        default_params: dict[str, Any] = {
            "objective": "regression",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "seed": 42,
            "num_threads": 8,
            "verbosity": -1,
        }
        if params is not None:
            default_params.update(params)
        self._params = default_params
        self._num_boost_round = num_boost_round
        super().__init__(None, categorical_mode="lightgbm")

    @property
    def params(self) -> dict[str, Any]:
        """Return the effective LightGBM parameters."""
        return dict(self._params)

    @property
    def num_boost_round(self) -> int:
        """Return the number of boosting rounds."""
        return self._num_boost_round

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
