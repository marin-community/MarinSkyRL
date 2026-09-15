"""Shared storage and runtime provenance helpers for Tinker reproductions."""

from __future__ import annotations

import json
from importlib.metadata import version
from typing import Protocol
from urllib.parse import urlparse


class ArtifactStorage(Protocol):
    def list_dir(self, prefix: str) -> list[str]: ...

    def write(self, path: str, data: bytes) -> None: ...


def runtime_versions(packages: tuple[str, ...]) -> dict[str, str]:
    return {package: version(package) for package in packages}


def write_json(storage: ArtifactStorage, path: str, payload: object) -> None:
    storage.write(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def claim_empty_output(storage: ArtifactStorage, manifest_path: str, payload: object) -> None:
    existing = storage.list_dir("")
    if existing:
        raise RuntimeError(f"output prefix must be empty; found {existing}")
    write_json(storage, manifest_path, payload)


def validate_output_uri(output_uri: str, *, option_name: str = "--output-uri") -> None:
    parsed = urlparse(output_uri)
    if parsed.scheme != "s3" or not parsed.netloc or parsed.path in {"", "/"}:
        raise ValueError(f"{option_name} must be a non-root durable prefix using s3://")
