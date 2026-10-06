"""Publish prepared/profiled Pivot datasets under content-addressed object-store paths."""

import argparse
import hashlib
import json
from posixpath import dirname
from infra.rl_data.pivot import file_sha256
from marinskyrl.resource_locator import join_resource_path
import shutil
from pathlib import Path

from marinskyrl.remote_io import filesystem_and_path, open_output_stream


def publish_artifacts(artifacts: Path, destination: str) -> str:
    """Verify local checksums, upload files, and publish the completion manifest last."""
    manifest_path = artifacts / "manifest.json"
    payload = manifest_path.read_bytes()
    manifest = json.loads(payload)
    for name, info in manifest["artifacts"].items():
        digest = file_sha256(artifacts / f"{name}.parquet")
        if digest != info["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    parent = manifest
    while "parent" in parent:
        parent = parent["parent"]
    domain = "swe" if "SWE" in parent["dataset"] else "terminal"
    identity = hashlib.sha256(payload).hexdigest()
    relative = f"prepared/{domain}/{parent['revision']}/{identity}"
    if "profile_revision" in manifest:
        relative = f"profiles/{domain}/{manifest['profile_policy'].split('/')[-1]}/{manifest['profile_revision']}/{identity}"
    uri = join_resource_path(destination, relative)
    filesystem, path = filesystem_and_path(join_resource_path(uri, "manifest.json"))
    if filesystem.exists(path):
        with filesystem.open(path, "rb") as stream:
            if stream.read() != payload:
                raise ValueError("An immutable artifact manifest already exists with different contents")
        return uri
    files = sorted(item for item in artifacts.iterdir() if item.is_file() and item.name != "manifest.json")
    for source in [*files, manifest_path]:
        target_path = join_resource_path(dirname(path), source.name)
        filesystem.makedirs(dirname(path), exist_ok=True)
        with source.open("rb") as stream, open_output_stream(filesystem, target_path) as target:
            shutil.copyfileobj(stream, target, length=8 * 1024 * 1024)
    return uri


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    print(publish_artifacts(args.artifacts, args.destination))


if __name__ == "__main__":
    main()
