"""Optuna-powered Bayesian hyperparameter tuning for LightGBM time-series models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from forecasting.evaluation.metrics import mae, rmse, rmsle
from forecasting.models.lightgbm import LightGBMModel

KEY_COLUMNS = ["date", "store_nbr", "family"]


@dataclass(frozen=True)
class FoldData:
    """Precomputed feature datasets for one rolling-origin evaluation fold."""

    forecast_origin: pd.Timestamp
    train_features: pd.DataFrame
    future_features: pd.DataFrame
    actuals: pd.DataFrame


@dataclass(frozen=True)
class TuningResult:
    """Container for the completed hyperparameter tuning study."""

    best_params: dict[str, Any]
    best_num_boost_round: int
    best_value: float
    best_trial_number: int
    study: optuna.Study
    trials_dataframe: pd.DataFrame


class LightGBMTuner:
    """Bayesian optimization over LightGBM hyperparameters on rolling folds."""

    def __init__(
        self,
        folds: list[FoldData],
        metric: str = "mean_rmsle",
        seed: int = 42,
    ) -> None:
        """Initialize tuner with precomputed rolling folds.

        Parameters
        ----------
        folds : list[FoldData]
            Precomputed training features, future known covariates, and
            ground-truth actuals for each evaluation fold.
        metric : str
            Aggregate metric to minimize (default: 'mean_rmsle').
        seed : int
            Reproducibility seed for samplers.
        """
        if not folds:
            raise ValueError("folds must not be empty")
        self._folds = folds
        self._metric = metric
        self._seed = seed

    def sample_hyperparameters(self, trial: optuna.Trial) -> tuple[dict[str, Any], int]:
        """Sample a candidate hyperparameter configuration."""
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.15, log=True)
        num_leaves = trial.suggest_int("num_leaves", 15, 63)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 50)
        feature_fraction = trial.suggest_float("feature_fraction", 0.6, 1.0)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
        num_boost_round = trial.suggest_int("num_boost_round", 40, 150)

        params: dict[str, Any] = {
            "objective": "regression",
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "min_child_samples": min_child_samples,
            "feature_fraction": feature_fraction,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": 1 if bagging_fraction < 1.0 else 0,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "seed": self._seed,
            "num_threads": 8,
            "verbosity": -1,
        }
        return params, num_boost_round

    def evaluate_configuration(
        self, params: dict[str, Any], num_boost_round: int
    ) -> dict[str, float]:
        """Fit and score one candidate configuration across all folds."""
        fold_rmsle: list[float] = []
        fold_mae: list[float] = []
        fold_rmse: list[float] = []

        for fold in self._folds:
            model = LightGBMModel(params=params, num_boost_round=num_boost_round)
            model.fit(fold.train_features)
            predictions = model.predict(fold.future_features)
            scored = predictions.merge(
                fold.actuals, on=KEY_COLUMNS, how="inner", validate="one_to_one"
            )
            if len(scored) != len(fold.actuals):
                raise ValueError(
                    f"Prediction count {len(scored)} does not match actual count "
                    f"{len(fold.actuals)}"
                )
            fold_rmsle.append(rmsle(scored["actual"], scored["prediction"]))
            fold_mae.append(mae(scored["actual"], scored["prediction"]))
            fold_rmse.append(rmse(scored["actual"], scored["prediction"]))

        return {
            "mean_rmsle": float(np.mean(fold_rmsle)),
            "median_rmsle": float(np.median(fold_rmsle)),
            "mean_mae": float(np.mean(fold_mae)),
            "mean_rmse": float(np.mean(fold_rmse)),
        }

    def optimize(
        self,
        n_trials: int = 15,
        timeout: float | None = None,
        study_name: str = "lightgbm-hyperparameter-tuning",
        mlflow_tracking: bool = False,
        project_root: Path | None = None,
    ) -> TuningResult:
        """Run the Bayesian optimization study."""
        sampler = optuna.samplers.TPESampler(seed=self._seed)
        study = optuna.create_study(
            direction="minimize",
            sampler=sampler,
            study_name=study_name,
        )

        def objective(trial: optuna.Trial) -> float:
            params, num_boost_round = self.sample_hyperparameters(trial)
            metrics_dict = self.evaluate_configuration(params, num_boost_round)
            for metric_name, metric_val in metrics_dict.items():
                trial.set_user_attr(metric_name, metric_val)

            if mlflow_tracking:
                self._log_trial_to_mlflow(
                    trial=trial,
                    params=params,
                    num_boost_round=num_boost_round,
                    metrics_dict=metrics_dict,
                    project_root=project_root,
                    study_name=study_name,
                )
            return metrics_dict[self._metric]

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = {
            k: v for k, v in study.best_trial.params.items() if k != "num_boost_round"
        }
        best_num_boost_round = int(study.best_trial.params.get("num_boost_round", 100))

        full_best_params: dict[str, Any] = {
            "objective": "regression",
            "seed": self._seed,
            "num_threads": 8,
            "verbosity": -1,
            **best_params,
        }
        if full_best_params.get("bagging_fraction", 1.0) < 1.0:
            full_best_params["bagging_freq"] = 1

        trials_df = study.trials_dataframe()
        return TuningResult(
            best_params=full_best_params,
            best_num_boost_round=best_num_boost_round,
            best_value=study.best_value,
            best_trial_number=study.best_trial.number,
            study=study,
            trials_dataframe=trials_df,
        )

    def _log_trial_to_mlflow(
        self,
        trial: optuna.Trial,
        params: dict[str, Any],
        num_boost_round: int,
        metrics_dict: dict[str, float],
        project_root: Path | None,
        study_name: str,
    ) -> None:
        """Log trial parameters and rolling-origin metrics to MLflow."""
        try:
            import mlflow

            if project_root:
                db_path = project_root / "mlflow.db"
                mlflow.set_tracking_uri(f"sqlite:///{db_path.resolve().as_posix()}")

            mlflow.set_experiment(study_name)
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                all_params = {**params, "num_boost_round": num_boost_round}
                mlflow.log_params(all_params)
                mlflow.log_metrics(metrics_dict)
                mlflow.set_tags(
                    {
                        "trial_number": str(trial.number),
                        "model_family": "lightgbm",
                        "tuning_engine": "optuna",
                    }
                )
        except Exception:
            pass  # Tracking errors must never abort optimization loop
