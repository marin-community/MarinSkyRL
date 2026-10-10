"""Supply Iris runtime identity for task-owned direct client submissions."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

from cloud.iris.runtime_bundle import BUNDLE_IDENTITY_FILE, read_manifest_paths, validate_bundled_runtime

SOURCE_MANIFEST = "hero-final-source-manifest.json"


def manifest_identity(root: Path) -> tuple[str, list[dict[str, str]]]:
    files = []
    for value in read_manifest_paths(root):
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Invalid runtime bundle path: {value}")
        files.append({"path": value, "sha256": hashlib.sha256((root / path).read_bytes()).hexdigest()})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return digest, files


def verify_source_manifest(
    root: Path, expected_commit: str, expected_sha256: str, *, verify_imports: bool = True
) -> int:
    path = root / SOURCE_MANIFEST
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(f"Source manifest digest mismatch: {actual_sha256} != {expected_sha256}")
    manifest = json.loads(path.read_text())
    if manifest["source_commit"] != expected_commit:
        raise ValueError("Source manifest commit mismatch")
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid source manifest path: {relative}")
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Source bundle file mismatch: {relative}")
    modules = ("skyrl_train.weight_sync.expert_block.driver", "skyrl_train.models.grug_moe", "cloud.iris.task_runtime")
    for module in modules if verify_imports else ():
        spec = importlib.util.find_spec(module)
        if spec is None or spec.origin is None or not Path(spec.origin).resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Running module is outside the pinned source bundle: {module}: {spec}")
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-digest", required=True)
    parser.add_argument("--source-manifest-sha256", required=True)
    args = parser.parse_args()
    root = Path.cwd()
    actual_digest, files = manifest_identity(root)
    if actual_digest != args.expected_digest:
        raise ValueError(f"Runtime bundle digest mismatch: {actual_digest} != {args.expected_digest}")
    (root / BUNDLE_IDENTITY_FILE).write_text(
        json.dumps({"launcher_commit": args.expected_commit, "files": files}, sort_keys=True) + "\n"
    )
    if validate_bundled_runtime(root) != args.expected_commit:
        raise ValueError("Runtime bundle launcher commit mismatch")
    count = verify_source_manifest(root, args.expected_commit, args.source_manifest_sha256)
    print(
        f"Verified runtime bundle {args.expected_commit} {actual_digest}; source files={count} manifest={args.source_manifest_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
