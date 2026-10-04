"""Unit tests for the feature-level inference API."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import forecasting.api.app as api


def valid_row() -> dict[str, object]:
    """Build one complete feature-level request row."""
    row: dict[str, object] = {feature: 1.0 for feature in api.MODEL_FEATURES}
    for feature in api.CATEGORICAL_FEATURES:
        row[feature] = "GROCERY I"
    row.update({"store_nbr": 1, "cluster": 1})
    return row


class FakeModel:
    """Simple native-model double."""

    pandas_categorical = [
        ["1"],
        ["GROCERY I"],
        ["GROCERY I"],
        ["GROCERY I"],
        ["GROCERY I"],
        ["1"],
    ]

    def predict(self, frame):
        assert list(frame.columns) == api.MODEL_FEATURES
        return [12.5] * len(frame)


def test_health_reports_loaded_candidate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Startup loads the candidate and health exposes its status."""
    monkeypatch.setattr(api, "load_registered_model", lambda: FakeModel())
    with TestClient(api.app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_name": "favorita-forecast-model",
        "model_alias": "candidate",
        "model_loaded": True,
    }


def test_predict_returns_numeric_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid prepared features are passed to the loaded model."""
    monkeypatch.setattr(api, "load_registered_model", lambda: FakeModel())
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"rows": [valid_row(), valid_row()]})
    assert response.status_code == 200
    assert response.json() == {"predictions": [12.5, 12.5]}


def test_invalid_request_schema_returns_422() -> None:
    """Missing feature rows are rejected by Pydantic."""
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"rows": [{"store_nbr": 1}]})
    assert response.status_code == 422


def test_model_load_failure_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    """A load failure leaves health available and prediction returns 503."""
    monkeypatch.setattr(
        api,
        "load_registered_model",
        lambda: (_ for _ in ()).throw(RuntimeError("registry unavailable")),
    )
    with TestClient(api.app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        assert client.post("/predict", json={"rows": [valid_row()]}).status_code == 503


def test_prediction_failure_returns_422(monkeypatch: pytest.MonkeyPatch) -> None:
    """Backend feature incompatibility is returned without a stack trace."""

    class FailingModel(FakeModel):
        def predict(self, frame):
            raise ValueError("bad feature")

    monkeypatch.setattr(api, "load_registered_model", lambda: FailingModel())
    with TestClient(api.app) as client:
        response = client.post("/predict", json={"rows": [valid_row()]})
    assert response.status_code == 422
    assert response.json()["detail"] == "Incompatible feature values: ValueError"


def test_predict_clips_negative_predictions_to_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative raw predictions are clipped to 0.0 for single and multi-row requests."""

    class NegativeModel(FakeModel):
        def predict(self, frame):
            return [-10.5, 5.0, -0.1] if len(frame) == 3 else [-15.0]

    monkeypatch.setattr(api, "load_registered_model", lambda: NegativeModel())
    with TestClient(api.app) as client:
        single_res = client.post("/predict", json={"rows": [valid_row()]})
        assert single_res.status_code == 200
        assert single_res.json() == {"predictions": [0.0]}

        multi_res = client.post(
            "/predict", json={"rows": [valid_row(), valid_row(), valid_row()]}
        )
        assert multi_res.status_code == 200
        assert multi_res.json() == {"predictions": [0.0, 5.0, 0.0]}
