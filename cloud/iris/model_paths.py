"""Iris-specific model-path helpers."""

from marinskyrl.resource_locator import ModelSource


def model_source_cli_args(model_source_uri: str | None, model_source_identity: str | None) -> list[str]:
    """Render a validated model source for Iris command-line boundaries."""
    if model_source_uri and model_source_identity is None:
        return ["--model-source-uri", model_source_uri]
    source = ModelSource.optional(model_source_uri, model_source_identity)
    if source is None:
        return []
    return ["--model-source-uri", source.uri, "--model-source-identity", source.identity]
