"""Build the launcher-revision runtime bundle synchronized into task images."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

BUNDLE_FILE_MANIFEST = Path("cloud/iris/runtime_bundle_files.txt")
BUNDLE_IDENTITY_FILE = ".marinskyrl-runtime.json"


@dataclass(frozen=True)
class LauncherSource:
    """Committed checkout selected as the source of launcher runtime files."""

    root: Path
    commit: str


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _checkout_root(path: Path) -> Path | None:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    root = Path(result.stdout.strip()).resolve()
    project_file = root / "pyproject.toml"
    if not project_file.is_file() or not (root / BUNDLE_FILE_MANIFEST).is_file():
        return None
    project = tomllib.loads(project_file.read_text()).get("project", {})
    return root if project.get("name") == "marinskyrl" else None


def _installed_checkout() -> Path | None:
    try:
        direct_url = importlib.metadata.distribution("marinskyrl").read_text("direct_url.json")
    except importlib.metadata.PackageNotFoundError:
        return None
    if not direct_url:
        return None
    parsed_url = urlparse(json.loads(direct_url).get("url", ""))
    if parsed_url.scheme != "file":
        return None
    checkout = Path(unquote(parsed_url.path)).resolve()
    return _checkout_root(checkout)


def resolve_launcher_source() -> LauncherSource:
    """Resolve the checkout whose committed runtime bytes a launch will ship."""
    checkout = _checkout_root(Path.cwd()) or _installed_checkout()
    if checkout is None:
        raise RuntimeError(
            "Cannot locate the MarinSkyRL checkout for this launcher. Run from the intended checkout or reinstall "
            "marinskyrl from it with `uv sync --frozen --group dev --extra cpu --extra telemetry "
            "--reinstall-package marinskyrl`."
        )
    return LauncherSource(root=checkout, commit=_git_output(checkout, "rev-parse", "HEAD"))


def _bundle_paths(source: LauncherSource) -> tuple[Path, ...]:
    manifest = source.root / BUNDLE_FILE_MANIFEST
    paths: list[Path] = []
    for line in manifest.read_text().splitlines():
        value = line.strip()
        if not value or value.startswith("#"):
            continue
        path = Path(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Runtime bundle path must stay below the checkout: {value}")
        if path in paths:
            raise ValueError(f"Runtime bundle manifest contains a duplicate path: {value}")
        if not (source.root / path).is_file():
            raise FileNotFoundError(f"Runtime bundle file does not exist: {value}")
        paths.append(path)
    if not paths:
        raise ValueError(f"Runtime bundle manifest is empty: {manifest}")
    return tuple(paths)


def _reject_uncommitted_runtime(source: LauncherSource, paths: tuple[Path, ...]) -> None:
    relative_paths = [BUNDLE_FILE_MANIFEST.as_posix(), *(path.as_posix() for path in paths)]
    status = _git_output(source.root, "status", "--porcelain", "--untracked-files=all", "--", *relative_paths)
    if status:
        raise RuntimeError(
            "Runtime bundle inputs contain uncommitted changes, so launcher_commit cannot identify their bytes:\n"
            f"{status}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_bundle_inputs(expected_launcher_commit: str) -> tuple[LauncherSource, tuple[Path, ...]]:
    source = resolve_launcher_source()
    if source.commit != expected_launcher_commit:
        raise ValueError(
            f"Selected launcher checkout commit {source.commit} does not match requested {expected_launcher_commit}"
        )
    paths = _bundle_paths(source)
    _reject_uncommitted_runtime(source, paths)
    return source, paths


def validate_runtime_bundle(expected_launcher_commit: str) -> None:
    """Reject a runtime bundle that cannot be identified by the requested commit."""
    _validated_bundle_inputs(expected_launcher_commit)


def build_runtime_bundle(expected_launcher_commit: str) -> Path:
    """Copy committed runtime files from the selected checkout into an Iris workspace."""
    source, paths = _validated_bundle_inputs(expected_launcher_commit)

    workspace = Path(tempfile.mkdtemp(prefix="marinskyrl-runtime-bundle-"))
    files: list[dict[str, str]] = []
    for relative_path in paths:
        destination = workspace / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source.root / relative_path, destination)
        files.append({"path": relative_path.as_posix(), "sha256": _sha256(destination)})
    (workspace / BUNDLE_IDENTITY_FILE).write_text(
        json.dumps({"launcher_commit": source.commit, "files": files}, indent=2, sort_keys=True) + "\n"
    )
    return workspace
