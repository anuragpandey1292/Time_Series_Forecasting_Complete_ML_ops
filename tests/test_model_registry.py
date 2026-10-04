"""Focused tests for MLflow Model Registry operations."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from forecasting.tracking.model_registry import register_model


def test_register_model_assigns_alias_and_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Registration links the source run and assigns candidate."""
    calls: list[tuple[object, ...]] = []

    class FakeClient:
        def __init__(self, tracking_uri=None) -> None:
            calls.append(("client", tracking_uri))

        def set_model_version_tag(self, *args) -> None:
            calls.append(("tag", *args))

        def set_registered_model_alias(self, *args) -> None:
            calls.append(("alias", *args))

    monkeypatch.setattr(
        "mlflow.register_model",
        lambda uri, name: SimpleNamespace(version="3"),
    )
    monkeypatch.setattr("mlflow.tracking.MlflowClient", FakeClient)

    result = register_model(
        run_id="run-123",
        artifact_path="model",
        registered_model_name="favorita-forecast-model",
        alias="candidate",
        tracking_uri="sqlite:///test.db",
        tags={"algorithm": "lightgbm"},
    )

    assert result.version == "3"
    assert result.model_uri == "runs:/run-123/model"
    assert ("alias", "favorita-forecast-model", "candidate", "3") in calls
    assert ("tag", "favorita-forecast-model", "3", "algorithm", "lightgbm") in calls


def test_register_model_propagates_invalid_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nonexistent source run is not silently registered."""

    def fail(*args, **kwargs):
        raise ValueError("source run does not exist")

    monkeypatch.setattr("mlflow.register_model", fail)
    with pytest.raises(ValueError, match="source run does not exist"):
        register_model(
            run_id="missing",
            artifact_path="model",
            registered_model_name="favorita-forecast-model",
        )
