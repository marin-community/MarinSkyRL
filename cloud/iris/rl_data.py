"""RL data staging and launch geometry checks."""

from __future__ import annotations

import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from marinskyrl.remote_io import filesystem_and_path
from marinskyrl.resource_locator import dataset_reference_parts, parse_hf_dataset_selector
from cloud.iris.hf_datasets import resolve_hf_dataset_selector
from marinskyrl.task_sources import DirectoryDataSource, data_source, data_source_dict


@dataclass(frozen=True)
class ResolvedRLData:
    """Runtime data entries paired with their immutable or local source references."""

    paths: tuple[str | dict[str, Any], ...]
    sources: tuple[str | dict[str, Any], ...]


def _stage_remote_file(uri: str, destination: Path, *, overwrite: bool) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    staging = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        filesystem, source_path = filesystem_and_path(uri)
        with filesystem.open(source_path, "rb") as src, open(staging, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(staging, destination)
    finally:
        staging.unlink(missing_ok=True)


def resolve_rl_train_data(
    train_data: list[str | dict[str, Any]],
    scratch_dir: str | None = None,
    on_exist: str = "skip",
    verbose: bool = True,
) -> ResolvedRLData:
    """Return staged files or pinned Hub selectors with their source references."""
    stage_root = Path(scratch_dir) / "rl_data" if scratch_dir else Path("/tmp/skyrl_rl_data")
    resolved: list[str] = []
    sources: list[str | dict[str, Any]] = []
    for entry in train_data:
        if isinstance(entry, Mapping):
            source = data_source(entry)
            if not isinstance(source, DirectoryDataSource):
                raise ValueError("Prepare packed task sources as TaskSpecs before launch")
            resolved.append(source.resolved_path())
            sources.append(data_source_dict(source))
            continue
        if "://" not in entry:
            name, split = dataset_reference_parts(entry)
            if not Path(entry).exists() and parse_hf_dataset_selector(name) is not None:
                pinned = resolve_hf_dataset_selector(name).canonical()
                entry = f"{pinned}:{split}" if split is not None else pinned
            resolved.append(entry)
            sources.append(entry)
            continue
        if entry.startswith("file://"):
            resolved.append(entry)
            sources.append(entry)
            continue
        identity = hashlib.sha256(entry.encode()).hexdigest()[:20]
        local = stage_root / identity / Path(entry.split("?", 1)[0]).name
        _stage_remote_file(entry, local, overwrite=on_exist == "overwrite")
        if verbose:
            print(f"[rl_data] Staged {entry} -> {local}")
        resolved.append(str(local))
        sources.append(entry)
    return ResolvedRLData(tuple(resolved), tuple(sources))


def compute_num_inference_engines(
    num_nodes: int,
    gpus_per_node: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Compute the number of vLLM inference engines: total GPUs // tensor_parallel_size."""
    total_gpus = num_nodes * gpus_per_node
    return total_gpus // tensor_parallel_size


def check_rl_environment() -> Optional[Path]:
    """Locate an optional standalone RL venv, or None.

    Checks ``$DCFT_RL_ENV``, ``$DCFT/envs/rl``, then ``<repo>/envs/rl``. Returns
    None when no such venv exists (the caller then uses ``sys.executable``, which
    on Iris is already the frozen task venv's Python).
    """
    candidates = []
    if os.environ.get("DCFT_RL_ENV"):
        candidates.append(Path(os.environ["DCFT_RL_ENV"]))
    if os.environ.get("DCFT"):
        candidates.append(Path(os.environ["DCFT"]) / "envs" / "rl")
    candidates.append(Path(__file__).resolve().parents[2] / "envs" / "rl")

    for candidate in candidates:
        if candidate.exists() and (candidate / "bin" / "activate").exists():
            return candidate

    return None
