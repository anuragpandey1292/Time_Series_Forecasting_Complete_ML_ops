"""Global CatBoost forecasting model with recursive target features."""

from catboost import CatBoostRegressor

from forecasting.models.global_ml import GlobalRecursiveRegressor


class CatBoostModel(GlobalRecursiveRegressor):
    """Fixed global CatBoost regressor for store-family daily forecasts."""

    def __init__(self) -> None:
        """Create the initial fixed RMSE configuration."""
        estimator = CatBoostRegressor(
            loss_function="RMSE",
            iterations=100,
            depth=6,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=8,
        )
        super().__init__(estimator, categorical_mode="catboost")
