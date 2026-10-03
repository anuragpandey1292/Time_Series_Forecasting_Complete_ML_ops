"""Last-observation baseline forecasting model."""

from __future__ import annotations

import pandas as pd

from forecasting.models.base import (
    PREDICTION_COLUMNS,
    SERIES_COLUMNS,
    prepare_future_data,
    prepare_train_data,
)


class NaiveModel:
    """Repeat each store-family series' latest observed sales value."""

    def __init__(self) -> None:
        """Create an unfitted naive forecaster."""
        self._last_observations: pd.DataFrame | None = None

    def fit(self, train_data: pd.DataFrame) -> None:
        """Store the latest historical sales observation for each series."""
        prepared = prepare_train_data(train_data)
        latest = prepared.sort_values("date").groupby(
            SERIES_COLUMNS, sort=False, as_index=False
        ).tail(1)
        self._last_observations = latest.loc[:, [*SERIES_COLUMNS, "sales"]].copy()

    def predict(self, future_data: pd.DataFrame) -> pd.DataFrame:
        """Repeat each fitted series value for the requested future rows."""
        if self._last_observations is None:
            raise RuntimeError("NaiveModel must be fitted before predict()")
        future = prepare_future_data(future_data)
        predictions = future.merge(
            self._last_observations,
            on=SERIES_COLUMNS,
            how="left",
            validate="many_to_one",
            sort=False,
        )
        if predictions["sales"].isna().any():
            unknown = predictions.loc[
                predictions["sales"].isna(), SERIES_COLUMNS
            ].drop_duplicates()
            raise ValueError(
                "No historical sales found for future series: "
                f"{unknown.to_dict(orient='records')}"
            )
        return predictions.rename(columns={"sales": "prediction"}).loc[
            :, PREDICTION_COLUMNS
        ]
