"""Iris-specific model-path helpers."""

from marinskyrl.resource_locator import ModelSource


def model_source_cli_args(model_source_uri: str | None, model_source_identity: str | None) -> list[str]:
    """Render a validated model source for Iris command-line boundaries."""
    source = ModelSource.optional(model_source_uri, model_source_identity)
    if source is None:
        return []
    return ["--model-source-uri", source.uri, "--model-source-identity", source.identity]


def unsupported_model_path_message(model_path: str) -> str:
    """Explain the supported model and warm-mirror contract."""
    return (
        "--model_path must be a Hugging Face repo ID or a task-local directory; "
        f"got unsupported object-store URI {model_path!r}. To use an object-store mirror, "
        "pass the Hugging Face repo ID as --model_path and the seeded S3 prefix as --model-warm-source."
    )
