"""Portable manifests for immutable Hugging Face model exports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct

from marinskyrl.hf_model import normalize_fast_tokenizer_metadata, validate_portable_hf_model_files

MODEL_MANIFEST_FILENAME = ".marinskyrl-model-manifest.json"
HF_WEIGHT_INDEX_FILENAME = "model.safetensors.index.json"


@dataclass(frozen=True)
class ModelManifestFile:
    path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    model_id: str | None
    revision: str | None
    identity: str
    files: tuple[ModelManifestFile, ...]
    format_version: int = 1

    @classmethod
    def from_mapping(cls, value: object, source: str) -> "ModelManifest":
        if not isinstance(value, dict):
            raise ValueError(f"Invalid model manifest at {source}")
        try:
            files = tuple(ModelManifestFile(**entry) for entry in value["files"])
            manifest = cls(
                model_id=value.get("model_id"),
                revision=value.get("revision"),
                identity=value["identity"],
                files=files,
                format_version=value["format_version"],
            )
        except (KeyError, TypeError) as error:
            raise ValueError(f"Invalid model manifest at {source}") from error
        manifest.validate(source)
        return manifest

    def validate(self, source: str) -> None:
        if (
            type(self.format_version) is not int
            or self.format_version != 1
            or not isinstance(self.identity, str)
            or not self.identity.startswith("sha256:")
            or not self.files
            or (self.model_id is not None and not isinstance(self.model_id, str))
            or (self.revision is not None and not isinstance(self.revision, str))
        ):
            raise ValueError(f"Invalid model manifest at {source}")
        if any(
            not isinstance(entry.path, str) or type(entry.size) is not int or not isinstance(entry.sha256, str)
            for entry in self.files
        ):
            raise ValueError(f"Invalid model manifest at {source}")
        paths = [entry.path for entry in self.files]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Model manifest contains duplicate paths: {source}")
        for entry in self.files:
            path = PurePosixPath(entry.path)
            invalid_checksum = len(entry.sha256) != 64 or any(
                character not in "0123456789abcdef" for character in entry.sha256
            )
            if (
                entry.path in ("", ".")
                or path.is_absolute()
                or ".." in path.parts
                or entry.size < 0
                or invalid_checksum
            ):
                raise ValueError(f"Invalid model manifest entry {entry!r}: {source}")
        expected = _manifest_identity(self.model_id, self.revision, self.files)
        if self.identity != expected:
            raise ValueError(f"Model manifest identity mismatch at {source}: {self.identity} != {expected}")
        validate_portable_hf_model_files(set(paths), source)
        if HF_WEIGHT_INDEX_FILENAME not in paths:
            raise ValueError(f"Model manifest is missing {HF_WEIGHT_INDEX_FILENAME}: {source}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_identity(
    model_id: str | None,
    revision: str | None,
    files: tuple[ModelManifestFile, ...],
) -> str:
    value = {
        "files": [asdict(entry) for entry in files],
        "format_version": 1,
        "model_id": model_id,
        "revision": revision,
    }
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _safetensors_keys(path: Path) -> tuple[str, ...]:
    with path.open("rb") as source:
        prefix = source.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {path}")
        header_size = struct.unpack("<Q", prefix)[0]
        header = json.loads(source.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")
    return tuple(sorted(key for key in header if key != "__metadata__"))


def _ensure_weight_index(snapshot: Path) -> None:
    shards = sorted(snapshot.glob("*.safetensors"))
    if not shards:
        raise ValueError(f"Hugging Face mirror requires safetensors weights: {snapshot}")
    weight_map: dict[str, str] = {}
    for shard in shards:
        for key in _safetensors_keys(shard):
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


def snapshot_model_manifest(snapshot: Path, model_id: str | None, revision: str | None) -> ModelManifest:
    normalize_fast_tokenizer_metadata(snapshot)
    _ensure_weight_index(snapshot)
    paths = sorted(
        path for path in snapshot.rglob("*") if path.is_file() and path.relative_to(snapshot).parts[0] != ".cache"
    )
    names = {path.relative_to(snapshot).as_posix() for path in paths}
    validate_portable_hf_model_files(names, str(snapshot))
    files = tuple(
        ModelManifestFile(
            path=path.relative_to(snapshot).as_posix(),
            size=path.stat().st_size,
            sha256=sha256_file(path),
        )
        for path in paths
    )
    return ModelManifest(
        model_id=model_id,
        revision=revision,
        identity=_manifest_identity(model_id, revision, files),
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
    marker.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")
    return manifest
