"""Portable manifests for immutable Hugging Face model exports."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
from typing import BinaryIO, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from marinskyrl.hf_model import (
    normalize_fast_tokenizer_metadata,
    validate_hf_model_weights,
    validate_portable_hf_model_files,
)

MODEL_MANIFEST_FILENAME = ".marinskyrl-model-manifest.json"
HF_WEIGHT_INDEX_FILENAME = "model.safetensors.index.json"


class ModelManifestFile(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    size: int = Field(ge=0, strict=True)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if value in ("", ".") or path.is_absolute() or ".." in path.parts:
            raise ValueError(f"model file path must be relative and contained: {value!r}")
        return value


class ModelManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_id: str | None
    revision: str | None
    tokenizer_mode: Literal["embedded", "policy"] = "embedded"
    identity: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    files: tuple[ModelManifestFile, ...] = Field(min_length=1)
    format_version: Literal[1] = 1

    @classmethod
    def from_mapping(cls, value: object, source: str) -> Self:
        try:
            return cls.model_validate(value, context={"source": source})
        except ValidationError as error:
            raise ValueError(f"Invalid model manifest at {source}: {error}") from error

    @model_validator(mode="after")
    def validate_manifest(self, info: ValidationInfo) -> Self:
        source = (info.context or {}).get("source", "model manifest")
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Model manifest contains duplicate paths: {source}")
        expected = _manifest_identity(self.model_id, self.revision, self.files, self.tokenizer_mode)
        if self.identity != expected:
            raise ValueError(f"Model manifest identity mismatch at {source}: {self.identity} != {expected}")
        names = set(paths)
        if self.tokenizer_mode == "embedded":
            validate_portable_hf_model_files(names, source)
        else:
            validate_hf_model_weights(names, source)
        if HF_WEIGHT_INDEX_FILENAME not in paths:
            raise ValueError(f"Model manifest is missing {HF_WEIGHT_INDEX_FILENAME}: {source}")
        return self


def sha256_stream(source: BinaryIO) -> str:
    digest = hashlib.sha256()
    while chunk := source.read(8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return sha256_stream(source)


def _manifest_identity(
    model_id: str | None,
    revision: str | None,
    files: tuple[ModelManifestFile, ...],
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> str:
    value = {
        "files": [entry.model_dump(mode="json") for entry in files],
        "format_version": 1,
        "model_id": model_id,
        "revision": revision,
    }
    # Preserve identities written before draft manifests declared their shared
    # policy-tokenizer contract. The non-default mode remains identity-bound.
    if tokenizer_mode != "embedded":
        value["tokenizer_mode"] = tokenizer_mode
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def safetensors_keys(source: BinaryIO, source_name: str) -> tuple[str, ...]:
    prefix = source.read(8)
    if len(prefix) != 8:
        raise ValueError(f"Truncated safetensors header: {source_name}")
    header_size = struct.unpack("<Q", prefix)[0]
    header = json.loads(source.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {source_name}")
    return tuple(sorted(key for key in header if key != "__metadata__"))


def _ensure_weight_index(snapshot: Path) -> None:
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"Hugging Face mirror requires safetensors weights: {snapshot}")
    weight_map: dict[str, str] = {}
    for shard in shards:
        with shard.open("rb") as source:
            keys = safetensors_keys(source, str(shard))
        for key in keys:
            if key in weight_map:
                raise ValueError(f"Duplicate tensor {key!r} in {snapshot}")
            weight_map[key] = shard.name
    if not weight_map:
        raise ValueError(f"Hugging Face mirror has no indexed tensors: {snapshot}")
    index = {
        "metadata": {"total_size": sum(path.stat().st_size for path in shards)},
        "weight_map": weight_map,
    }
    index_path = snapshot / HF_WEIGHT_INDEX_FILENAME
    if index_path.is_file():
        existing = json.loads(index_path.read_text())
        if existing.get("weight_map") != weight_map:
            raise ValueError(f"Safetensors weight index does not match its shards: {index_path}")
    else:
        index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")


def snapshot_model_manifest(
    snapshot: Path,
    model_id: str | None,
    revision: str | None,
    *,
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> ModelManifest:
    normalize_fast_tokenizer_metadata(snapshot)
    _ensure_weight_index(snapshot)
    paths = sorted(
        path for path in snapshot.rglob("*") if path.is_file() and path.relative_to(snapshot).parts[0] != ".cache"
    )
    names = {path.relative_to(snapshot).as_posix() for path in paths}
    if tokenizer_mode == "embedded":
        validate_portable_hf_model_files(names, str(snapshot))
    else:
        validate_hf_model_weights(names, str(snapshot))
    files = tuple(
        ModelManifestFile(
            path=path.relative_to(snapshot).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in paths
    )
    return model_manifest(
        files,
        model_id=model_id,
        revision=revision,
        tokenizer_mode=tokenizer_mode,
    )


def model_manifest(
    files: tuple[ModelManifestFile, ...],
    *,
    model_id: str | None,
    revision: str | None,
    tokenizer_mode: Literal["embedded", "policy"] = "embedded",
) -> ModelManifest:
    """Build and validate a manifest from an existing file inventory."""
    return ModelManifest(
        model_id=model_id,
        revision=revision,
        tokenizer_mode=tokenizer_mode,
        identity=_manifest_identity(model_id, revision, files, tokenizer_mode),
        files=files,
    )


def write_local_model_manifest(
    model_dir: str | Path,
    *,
    model_id: str | None = None,
    revision: str | None = None,
) -> ModelManifest:
    """Validate a completed local HF export and write its immutable manifest."""
    root = Path(model_dir)
    marker = root / MODEL_MANIFEST_FILENAME
    if marker.exists():
        marker.unlink()
    manifest = snapshot_model_manifest(root, model_id, revision)
    marker.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n")
    return manifest
