"""Small MLflow Model Registry operations for deployable model artifacts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegisteredModelVersion:
    """Identify a registered version and its source run."""

    name: str
    version: str
    run_id: str
    model_uri: str


def register_model(
    *,
    run_id: str,
    artifact_path: str,
    registered_model_name: str,
    alias: str | None = None,
    tracking_uri: str | None = None,
    tags: dict[str, str] | None = None,
) -> RegisteredModelVersion:
    """Register a run artifact and optionally assign an alias."""
    import mlflow
    from mlflow.tracking import MlflowClient

    if tracking_uri is not None:
        mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"runs:/{run_id}/{artifact_path}"
    registered = mlflow.register_model(model_uri, registered_model_name)
    client = MlflowClient(tracking_uri=tracking_uri)
    if tags:
        for key, value in tags.items():
            client.set_model_version_tag(
                registered_model_name, registered.version, key, str(value)
            )
    if alias:
        client.set_registered_model_alias(
            registered_model_name, alias, registered.version
        )
    return RegisteredModelVersion(
        name=registered_model_name,
        version=str(registered.version),
        run_id=run_id,
        model_uri=model_uri,
    )


def get_model_version_by_alias(
    registered_model_name: str,
    alias: str,
    tracking_uri: str | None = None,
) -> object:
    """Return the registered version currently pointed to by an alias."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    return client.get_model_version_by_alias(registered_model_name, alias)


def list_model_versions(
    registered_model_name: str,
    tracking_uri: str | None = None,
) -> list[object]:
    """List all versions of a registered model."""
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=tracking_uri)
    return list(client.search_model_versions(f"name='{registered_model_name}'"))
