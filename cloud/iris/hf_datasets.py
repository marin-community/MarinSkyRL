"""HuggingFace dataset helpers for the Iris RL launcher.

Detects the on-disk task-dataset format (raw task dirs versus parquet with
``task_binary``) and downloads dataset snapshots.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download


def is_raw_tasks_directory(snapshot_dir) -> bool:
    """Check if a directory contains raw task folders (not parquet with task_binary).

    Raw task directories have subdirectories with ``instruction.md`` files, rather
    than parquet files with ``task_binary`` columns that need extraction. Used to
    auto-detect the downloaded HuggingFace dataset format.
    """
    snapshot_dir = Path(snapshot_dir)

    parquet_files = list(snapshot_dir.rglob("*.parquet"))
    if parquet_files:
        try:
            import pyarrow.parquet as pq

            for pf in parquet_files[:1]:
                table = pq.read_table(pf)
                if "task_binary" in table.column_names:
                    return False  # needs extraction
        except Exception:
            pass

    instruction_files = list(snapshot_dir.rglob("instruction.md"))
    if instruction_files:
        return True

    return False


def download_hf_dataset(repo_id: str, revision: Optional[str] = None) -> str:
    """Download a HuggingFace dataset repo snapshot and return its local path."""
    if not repo_id or not isinstance(repo_id, str):
        raise ValueError("repo_id must be a non-empty string")
    cache_path = os.environ.get("HF_CACHE_DIR", os.path.expanduser("~/.cache/huggingface/hub"))
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=str(cache_path),
        revision=revision,
        repo_type="dataset",
    )
