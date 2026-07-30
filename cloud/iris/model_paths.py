"""Model-path contracts shared by the Iris RL launcher and controller."""

from urllib.parse import urlparse

OBJECT_STORE_MODEL_SCHEMES = frozenset({"s3", "gs", "gcs"})


def is_object_store_model_path(model_path: str) -> bool:
    """Return whether a model path uses an unsupported object-store scheme."""
    return urlparse(model_path).scheme.lower() in OBJECT_STORE_MODEL_SCHEMES


def unsupported_model_path_message(model_path: str) -> str:
    """Explain the supported model and warm-mirror contract."""
    return (
        "--model_path must be a Hugging Face repo ID or a task-local directory; "
        f"got unsupported object-store URI {model_path!r}. To use an object-store mirror, "
        "pass the Hugging Face repo ID as --model_path and the seeded S3 prefix as --model-warm-source."
    )
