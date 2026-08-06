"""RL job data-staging + topology helpers for the Iris launcher.

- ``resolve_rl_train_data``: extract HF task datasets to local task directories.
- ``compute_num_inference_engines`` / ``derive_skyrl_export_path``: placement math.
- ``check_rl_environment``: locate an optional standalone RL venv.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from cloud.iris.hf_datasets import is_hf_dataset_path


def resolve_rl_train_data(
    train_data: List[str],
    scratch_dir: Optional[str] = None,
    on_exist: str = "skip",
    verbose: bool = True,
    kind: str = "tasks",
) -> List[str]:
    """Resolve train_data paths for the configured data kind.

    ``kind="tasks"`` (default, terminal_bench agentic RL): SkyRL's
    TerminalBenchTaskDataset expects local directory paths where each subdirectory is a
    task containing an ``instruction.md`` file. HuggingFace dataset identifiers are
    extracted to ``$SCRATCH/tasks/<repo-name>/`` via
    ``cloud.iris.extract_tasks_from_parquet``, permissions fixed, and local paths returned.

    ``kind="parquet"`` (single-turn RLVR, e.g. main_base + aime): the entries are
    SkyRL-shaped parquet paths that ``PromptDataset`` loads via ``datasets.load_dataset``.
    Never task-extracted. Local paths and HF ids pass through UNCHANGED; an object-store URI
    (``s3://`` / ``gs://`` / http(s)) is STAGED to node-local disk and the local path returned —
    ``datasets.load_dataset`` refuses a remote URI under ``HF_HUB_OFFLINE=1`` (the mode set so
    the prestaged, template-rewritten model reads its warm cache), raising
    ``OfflineModeIsEnabled`` before training. Staging pulls it via fsspec (the pod's object-store
    creds + endpoint) so the offline flag only gates weights, not the dataset read.
    """
    if not train_data:
        return []

    if kind == "parquet":
        resolved: List[str] = []
        stage_root = Path(scratch_dir) / "rl_parquet" if scratch_dir else Path("/tmp/skyrl_rl_parquet")
        for entry in train_data:
            # A local path or a bare HF dataset id is read directly by PromptDataset.
            if "://" not in entry or entry.startswith("file://"):
                resolved.append(entry)
                continue
            # Object-store URI: pull to node-local disk (offline mode blocks the remote read).
            import fsspec

            stage_root.mkdir(parents=True, exist_ok=True)
            local = stage_root / Path(entry.split("?", 1)[0]).name
            if not local.exists() or on_exist == "overwrite":
                with fsspec.open(entry, "rb") as src, open(local, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            if verbose:
                print(f"[rl_data] data.kind=parquet: staged {entry} -> {local}")
            resolved.append(str(local))
        return resolved

    # Determine the scratch directory for extracted tasks. It MUST be a shared
    # filesystem visible to all compute nodes; /tmp is node-local (last resort).
    if scratch_dir is None:
        for env_var in ["SCRATCH", "DCFT", "DCFT_PRIVATE", "HOME"]:
            if os.environ.get(env_var):
                scratch_dir = os.environ[env_var]
                break
        else:
            scratch_dir = "/tmp"
            print(
                "[rl_data] WARNING: Using /tmp for task extraction. "
                "This is local to each node and may fail on multi-node jobs. "
                "Set $SCRATCH, $DCFT, or $DCFT_PRIVATE to a shared filesystem path."
            )
    tasks_base = Path(scratch_dir) / "tasks"

    resolved_paths = []

    for data_path in train_data:
        if is_hf_dataset_path(data_path):
            repo_name = data_path.split("/")[-1]
            output_dir = tasks_base / repo_name

            if verbose:
                print(f"[rl_data] Extracting HF dataset: {data_path}")
                print(f"[rl_data] Output directory: {output_dir}")

            if on_exist == "skip" and output_dir.exists() and any(output_dir.iterdir()):
                if verbose:
                    print(f"[rl_data] Tasks already extracted, skipping: {output_dir}")
                resolved_paths.append(str(output_dir))
                continue

            cmd = [
                sys.executable,
                "-m",
                "cloud.iris.extract_tasks_from_parquet",
                "--parquet",
                data_path,
                "--output_dir",
                str(output_dir),
                "--on_exist",
                on_exist,
            ]

            if verbose:
                print(f"[rl_data] Running: {' '.join(cmd)}")

            # A stalled HF download inside the extractor (mid-download socket hang)
            # would block subprocess.run forever, so a per-attempt timeout converts
            # a stall into a retry; HF resumes the partial `.incomplete` shard on the
            # next attempt, so a killed-mid-download attempt loses nothing. 600s
            # covers a clean extract yet fits several retries inside a gang-join budget.
            extract_attempt_timeout = 600
            last_err = ""
            for attempt in range(1, 7):
                try:
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True,
                        timeout=extract_attempt_timeout,
                    )
                    if verbose and result.stdout:
                        print(result.stdout)
                    break
                except subprocess.TimeoutExpired:
                    last_err = (
                        f"extract stalled > {extract_attempt_timeout}s "
                        "(mid-download socket hang); killed, retrying (HF resumes the partial shard)"
                    )
                    print(f"[rl_data] extract attempt {attempt}/6 TIMED OUT for {data_path}: {last_err}")
                    time.sleep(min(30, 2**attempt))
                except subprocess.CalledProcessError as e:
                    last_err = f"stdout: {e.stdout}\n  stderr: {e.stderr}"
                    print(f"[rl_data] extract attempt {attempt}/6 failed for {data_path}:")
                    print(f"  {last_err}")
                    time.sleep(min(30, 2**attempt))
            else:
                raise RuntimeError(f"Failed to extract HF dataset after 6 attempts: {data_path}: {last_err}")

            _fix_task_permissions(output_dir, verbose=verbose)
            resolved_paths.append(str(output_dir))
        else:
            local_path = Path(data_path)
            if local_path.exists():
                _fix_task_permissions(local_path, verbose=verbose)
            resolved_paths.append(data_path)

    return resolved_paths


def _fix_task_permissions(task_dir: Path, verbose: bool = True) -> None:
    """Run ``chmod -R a+rX`` on a task tree so files are readable + dirs traversable.

    Idempotency guard: a single cheap ``stat`` of the top-level dir short-circuits
    when its perms already satisfy ``a+rX``, avoiding a ~100K-write recursive chmod
    over an already-correct tree (which has wedged the launcher under GPFS metadata
    contention). Any dir that genuinely needs the fix still gets the full recursive
    chmod. Never uses find/du.
    """
    if not task_dir.exists():
        return

    rx_bits = stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH
    try:
        mode = task_dir.stat().st_mode
        if (mode & rx_bits) == rx_bits:
            if verbose:
                print(f"[rl_data] Permissions already a+rX on top-level dir, skipping recursive chmod: {task_dir}")
            return
    except OSError as e:
        if verbose:
            print(f"[rl_data] stat probe failed on {task_dir} ({e}); running recursive chmod to be safe.")

    if verbose:
        print(f"[rl_data] Fixing permissions on: {task_dir}")

    try:
        subprocess.run(
            ["chmod", "-R", "a+rX", str(task_dir)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        # Don't fail the whole job for permission issues.
        print(f"[rl_data] Warning: chmod failed on {task_dir}: {e.stderr}")


def compute_num_inference_engines(
    num_nodes: int,
    gpus_per_node: int,
    tensor_parallel_size: int = 1,
) -> int:
    """Compute the number of vLLM inference engines: total GPUs // tensor_parallel_size."""
    total_gpus = num_nodes * gpus_per_node
    return total_gpus // tensor_parallel_size


def derive_skyrl_export_path(
    experiments_dir: str,
    run_name: str,
    exports_subdir: str = "exports",
) -> str:
    """Derive the SkyRL export path (``<experiments_dir>/<run_name>/<exports_subdir>``)."""
    return str(Path(experiments_dir) / run_name / exports_subdir)


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
