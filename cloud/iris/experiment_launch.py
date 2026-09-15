"""Shared, secret-free launch mechanics for bounded Iris experiments."""

from __future__ import annotations

import re
import shlex
import subprocess
from pathlib import Path

_DIGEST_ADDRESSED_IMAGE = re.compile(r"[^@]+@sha256:[0-9a-f]{64}")


def validate_digest_addressed_image(task_image: str) -> None:
    if not _DIGEST_ADDRESSED_IMAGE.fullmatch(task_image):
        raise ValueError("--task-image must be a digest-addressed image reference")


def iris_task_command(
    *,
    cluster_config: Path,
    gpu: str,
    cpu: str | int,
    memory: str,
    disk: str,
    priority: str,
    task_image: str,
    job_name: str,
    task_module: str,
    task_args: tuple[str, ...],
) -> tuple[str, ...]:
    """Build the common single-slice, non-retrying Iris experiment command."""
    return (
        "uv",
        "run",
        "--frozen",
        "iris",
        "--config",
        str(cluster_config.resolve()),
        "job",
        "run",
        "--enable-extra-resources",
        "--gpu",
        gpu,
        "--cpu",
        str(cpu),
        "--memory",
        memory,
        "--disk",
        disk,
        "--priority",
        priority,
        "--no-preemptible",
        "--max-retries",
        "0",
        "--task-image",
        task_image,
        "--no-sync",
        "--no-wait",
        "--job-name",
        job_name,
        "--",
        "python",
        "-m",
        task_module,
        *task_args,
    )


def submit_or_print(command: tuple[str, ...], *, submit: bool, reviewed: bool, review_option: str) -> int:
    """Print a command and require its experiment-specific review gate before submission."""
    print(shlex.join(command))
    if not submit:
        print(f"Dry run only. Add --submit {review_option} after reviewing the plan.")
        return 0
    if not reviewed:
        raise SystemExit(f"--submit requires {review_option}")
    return subprocess.run(command, check=False).returncode
