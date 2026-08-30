"""Durable resource-reference contracts shared by training and launch tooling."""

from dataclasses import dataclass
import os
import posixpath
from urllib.parse import quote, urlsplit, urlunsplit

CLOUD_URI_SCHEMES = frozenset({"s3", "gs", "gcs"})
CLOUD_URI_PREFIXES = tuple(f"{scheme}://" for scheme in sorted(CLOUD_URI_SCHEMES))
HF_SELECTOR_REVISION_SEPARATOR = "@"
HF_SELECTOR_SUBDIR_SEPARATOR = "::"


class ModelLocatorError(ValueError):
    """A serialized or CLI model locator cannot be resolved unambiguously."""


def is_cloud_uri(path: str) -> bool:
    """Return whether a path uses a supported object-store scheme."""
    return path.startswith(CLOUD_URI_PREFIXES)


def join_resource_path(root: str, *parts: str) -> str:
    """Join local or object-store path components without corrupting URI syntax."""
    if not is_cloud_uri(root):
        return os.path.join(root, *parts)
    parsed = urlsplit(root)
    joined_path = posixpath.join(parsed.path, *parts)
    return urlunsplit((parsed.scheme, parsed.netloc, joined_path, parsed.query, parsed.fragment))


def is_hugging_face_repo_id(repo_id: str) -> bool:
    """Return whether a value has Hugging Face's ``namespace/repository`` form."""
    if not repo_id or repo_id.count("/") != 1 or HF_SELECTOR_SUBDIR_SEPARATOR in repo_id:
        return False
    if repo_id.startswith(("./", "../", "/", "~")) or "\\" in repo_id:
        return False
    return all(part.strip() not in ("", ".", "..") for part in repo_id.split("/"))


@dataclass(frozen=True)
class HFDatasetSelector:
    """A Hugging Face dataset repository with an optional revision and subdirectory."""

    repo_id: str
    revision: str | None = None
    subdir: str | None = None

    def canonical(self) -> str:
        revision = f"{HF_SELECTOR_REVISION_SEPARATOR}{self.revision}" if self.revision else ""
        subdir = f"{HF_SELECTOR_SUBDIR_SEPARATOR}{self.subdir}" if self.subdir else ""
        return f"{self.repo_id}{revision}{subdir}"

    def cache_name(self) -> str:
        """Return a reversible cache key that includes every selector component."""
        components = (("repo", self.repo_id), ("subdir", self.subdir), ("revision", self.revision))
        encoded = ((name, quote(value, safe="-._")) for name, value in components if value is not None)
        return "__".join(f"{name}-{len(value)}-{value}" for name, value in encoded)


def parse_hf_dataset_selector(value: str) -> HFDatasetSelector | None:
    """Parse ``org/repo[@revision][::subdir]`` without accepting local paths."""
    if not value or value.startswith(("./", "../", "/", "~")) or "\\" in value:
        return None
    repo_revision, subdir_separator, subdir = value.partition(HF_SELECTOR_SUBDIR_SEPARATOR)
    repo_id, revision_separator, revision = repo_revision.partition(HF_SELECTOR_REVISION_SEPARATOR)
    if not is_hugging_face_repo_id(repo_id):
        return None
    if revision_separator and not revision:
        return None
    if subdir_separator and (not subdir or subdir.startswith("/") or ".." in subdir.split("/")):
        return None
    return HFDatasetSelector(
        repo_id=repo_id,
        revision=revision if revision_separator else None,
        subdir=subdir if subdir_separator else None,
    )


@dataclass(frozen=True)
class ModelSource:
    """Immutable object-store source for a task-local model materialization."""

    uri: str
    identity: str

    def __post_init__(self) -> None:
        if not self.uri or not self.identity:
            raise ModelLocatorError("model_source_uri and model_source_identity must be provided together")
        if not is_cloud_uri(self.uri):
            raise ModelLocatorError(f"model_source_uri must be an object-store URI, got {self.uri!r}")

    @classmethod
    def optional(cls, uri: str | None, identity: str | None) -> "ModelSource | None":
        """Parse an optional source at a CLI or serialized-config boundary."""
        if uri is None and identity is None:
            return None
        return cls(uri=uri or "", identity=identity or "")


def model_source_for_path(
    model_path: str,
    model_source_uri: str | None,
    model_source_identity: str | None,
) -> ModelSource | None:
    """Parse a source and reject ambiguous Hub-ID plus object-store combinations."""
    source = ModelSource.optional(model_source_uri, model_source_identity)
    if source and is_hugging_face_repo_id(model_path):
        raise ModelLocatorError("model_source_uri requires a task-local model_path, not a Hugging Face repo ID")
    return source


def validate_replayable_model_reference(
    model_path: str,
    model_source_uri: str | None,
    model_source_identity: str | None,
) -> None:
    """Reject model locators that a later task cannot reconstruct."""
    source = model_source_for_path(model_path, model_source_uri, model_source_identity)
    if is_hugging_face_repo_id(model_path) or source:
        return
    raise ModelLocatorError(
        f"task-local model_path {model_path!r} requires model_source_uri and model_source_identity "
        "so a later export job can materialize it"
    )
