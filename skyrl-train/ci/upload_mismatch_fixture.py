"""Write and verify an immutable tiny-Grug mismatch fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath
from tests.gpu.tiny_grug import write_tiny_probe_fixture

COPY_BLOCK_BYTES = 8 * 1024 * 1024


def upload_tiny_probe_fixture(path: Path, prefix: str) -> dict[str, str]:
    """Copy and verify the fixture, returning SHA-256 digests by relative file path."""
    root = StoragePath(prefix)
    if (root / "fixture-sha256.json").isfile():
        raise FileExistsError(f"fixture prefix already has a completed upload: {prefix}")
    digests = {}
    for local in sorted(path.rglob("*")):
        if not local.is_file():
            continue
        relative = local.relative_to(path).as_posix()
        remote = root / relative
        local_hash = hashlib.sha256()
        with local.open("rb") as source, remote.open("wb") as target:
            for block in iter(lambda: source.read(COPY_BLOCK_BYTES), b""):
                local_hash.update(block)
                target.write(block)
        remote_hash = hashlib.sha256()
        with remote.open("rb") as source:
            for block in iter(lambda: source.read(COPY_BLOCK_BYTES), b""):
                remote_hash.update(block)
        if local_hash.digest() != remote_hash.digest():
            raise ValueError(f"uploaded fixture file differs: {relative}")
        digests[relative] = local_hash.hexdigest()
    (root / "fixture-sha256.json").write_text(json.dumps(digests, sort_keys=True) + "\n")
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-prefix")
    args = parser.parse_args()
    write_tiny_probe_fixture(args.output)
    if args.storage_prefix:
        digests = upload_tiny_probe_fixture(args.output, args.storage_prefix)
        print(json.dumps({"result": "PASS", "files": len(digests), "prefix": args.storage_prefix}, sort_keys=True))


if __name__ == "__main__":
    main()
