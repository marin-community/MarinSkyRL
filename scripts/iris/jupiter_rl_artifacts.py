"""Bounded artifact sync for explicitly selected JSC Jupiter RL runs.

Jupiter experiment data lives on GPFS. This module therefore requires the
canonical experiment directory and only touches known subtrees beneath it. It
never searches a scratch filesystem for jobs. Trace discovery is one shallow
listing of the run's standard ``trace_jobs`` directory, capped before transfer.
"""

from __future__ import annotations

import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol


LOG_SUFFIXES = (".log", ".out", ".err", ".jsonl", ".txt")
_HOST_PATTERN = re.compile(r"[A-Za-z0-9_.-]+")


class CommandRunner(Protocol):
    def __call__(
        self,
        arguments: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]: ...


@dataclass(frozen=True)
class JupiterRunSpec:
    """One Slurm allocation and its canonical launcher experiment directory."""

    job_id: str
    experiment_dir: str
    launcher_job_name: str | None = None

    @property
    def job_name(self) -> str:
        return self.launcher_job_name or PurePosixPath(self.experiment_dir).name

    @property
    def trace_root(self) -> str:
        return str(PurePosixPath(self.experiment_dir) / self.job_name / "trace_jobs")


@dataclass(frozen=True)
class JupiterArtifactSyncResult:
    finelog: str
    slurm_logs: str
    ray_logs: str
    traces: str
    trace_selected: int
    errors: tuple[str, ...]


@dataclass(frozen=True)
class JupiterJobStatus:
    state: str
    job_name: str
    error: str | None = None


_SLURM_STATES = {
    "PENDING": "pending",
    "CONFIGURING": "building",
    "RUNNING": "running",
    "COMPLETING": "running",
    "COMPLETED": "succeeded",
    "CANCELLED": "killed",
    "FAILED": "failed",
    "TIMEOUT": "failed",
    "OUT_OF_MEMORY": "failed",
    "NODE_FAIL": "failed",
    "PREEMPTED": "failed",
}


def parse_jupiter_run_spec(value: str) -> JupiterRunSpec:
    """Parse ``SLURM_JOB_ID=/absolute/experiment/path`` without path inference."""
    try:
        job_id, experiment_dir = value.split("=", 1)
    except ValueError as error:
        raise ValueError("--jupiter-run must be SLURM_JOB_ID=/absolute/experiment/path") from error
    if not job_id.isdigit():
        raise ValueError("Jupiter Slurm job ids must contain only digits")
    if any(character in experiment_dir for character in ("\0", "\n", "\r")):
        raise ValueError("Jupiter experiment paths may not contain control characters")
    path = PurePosixPath(experiment_dir)
    if not path.is_absolute() or str(path) in {"/", "."}:
        raise ValueError("Jupiter experiment path must be a specific absolute directory")
    if ".." in path.parts:
        raise ValueError("Jupiter experiment path may not traverse through '..'")
    return JupiterRunSpec(job_id, str(path))


def _run(
    runner: CommandRunner,
    arguments: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    return runner(arguments, capture_output=True, text=True, timeout=timeout)


def _ssh(
    runner: CommandRunner,
    host: str,
    command: str,
    *,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return _run(runner, ["ssh", "-o", "BatchMode=yes", host, command], timeout=timeout)


def _remote_exists(runner: CommandRunner, host: str, remote_path: str, *, directory: bool) -> bool:
    predicate = "-d" if directory else "-f"
    result = _ssh(runner, host, f"test {predicate} -- {shlex.quote(remote_path)}")
    return result.returncode == 0


def _finelog_path(runner: CommandRunner, host: str, logs_dir: str, job_id: str) -> str | None:
    """Resolve one Slurm stdout by job id with a shallow listing of ``logs/``."""
    result = _ssh(
        runner,
        host,
        f"ls -1t -- {shlex.quote(logs_dir)}/*_{job_id}.out 2>/dev/null | head -n 1",
    )
    candidate_text = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not candidate_text:
        return None
    candidate = PurePosixPath(candidate_text)
    if candidate.parent != PurePosixPath(logs_dir) or not candidate.name.endswith(f"_{job_id}.out"):
        return None
    return str(candidate)


def query_jupiter_job_status(
    run: JupiterRunSpec,
    *,
    host: str,
    runner: CommandRunner = subprocess.run,
) -> JupiterJobStatus:
    """Read one explicit Slurm job without enumerating the Jupiter queue."""
    if _HOST_PATTERN.fullmatch(host) is None:
        raise ValueError("--jupiter-host must be an SSH host name or configured alias")
    commands = (
        f"squeue -h -j {run.job_id} -o '%T|%j'",
        f"sacct -n -X -j {run.job_id} --format=State,JobName --parsable2 | head -n 1",
    )
    messages: list[str] = []
    for command in commands:
        result = _ssh(runner, host, command)
        line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
        if result.returncode == 0 and "|" in line:
            raw_state, job_name = (part.strip() for part in line.split("|", 1))
            normalized_state = raw_state.split("+", 1)[0].upper()
            return JupiterJobStatus(
                _SLURM_STATES.get(normalized_state, normalized_state.lower()),
                job_name.rstrip("|") or run.job_name,
            )
        message = (result.stderr or result.stdout).strip()
        if message:
            messages.append(message[-160:])
    detail = "; ".join(messages) or "job was absent from squeue and sacct"
    return JupiterJobStatus("unknown", run.job_name, detail)


def _rsync(
    runner: CommandRunner,
    host: str,
    remote_path: str,
    destination: Path,
    *,
    extra_arguments: tuple[str, ...] = (),
    timeout: int = 900,
) -> subprocess.CompletedProcess[str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    return _run(
        runner,
        [
            "rsync",
            "-az",
            "--partial",
            "--protect-args",
            *extra_arguments,
            "--",
            f"{host}:{remote_path}",
            str(destination),
        ],
        timeout=timeout,
    )


def _newest_trace_directories(
    runner: CommandRunner,
    host: str,
    trace_root: str,
    limit: int,
) -> tuple[str, ...]:
    quoted_root = shlex.quote(trace_root)
    limit_clause = "" if limit == 0 else f" | head -n {limit}"
    result = _ssh(
        runner,
        host,
        f"ls -1dt -- {quoted_root}/*/ 2>/dev/null{limit_clause}",
        timeout=300,
    )
    if result.returncode != 0 and not result.stdout.strip():
        return ()

    root = PurePosixPath(trace_root)
    selected: list[str] = []
    for line in result.stdout.splitlines():
        candidate = PurePosixPath(line.strip())
        try:
            relative = candidate.relative_to(root)
        except ValueError:
            continue
        if len(relative.parts) != 1:
            continue
        selected.append(str(candidate))
    return tuple(selected)


def _sync_trace_directory(
    runner: CommandRunner,
    host: str,
    remote_directory: str,
    destination: Path,
    max_non_log_bytes: int,
) -> tuple[str, ...]:
    errors: list[str] = []
    source = f"{remote_directory.rstrip('/')}/"
    destination.mkdir(parents=True, exist_ok=True)
    size_arguments = () if max_non_log_bytes == 0 else (f"--max-size={max_non_log_bytes}",)
    result = _rsync(runner, host, source, destination, extra_arguments=size_arguments)
    if result.returncode:
        errors.append((result.stderr or result.stdout).strip()[-240:])

    if max_non_log_bytes:
        log_filters = ("--include=*/", *(f"--include=*{suffix}" for suffix in LOG_SUFFIXES), "--exclude=*")
        result = _rsync(runner, host, source, destination, extra_arguments=log_filters)
        if result.returncode:
            errors.append((result.stderr or result.stdout).strip()[-240:])
    return tuple(error for error in errors if error)


def sync_jupiter_artifacts(
    run: JupiterRunSpec,
    destination: Path,
    *,
    host: str,
    trace_sync_limit: int,
    max_non_log_bytes: int,
    status_only: bool = False,
    runner: CommandRunner = subprocess.run,
) -> JupiterArtifactSyncResult:
    """Incrementally sync one explicit Jupiter experiment without broad GPFS scans."""
    if _HOST_PATTERN.fullmatch(host) is None:
        raise ValueError("--jupiter-host must be an SSH host name or configured alias")
    if trace_sync_limit < 0:
        raise ValueError("Jupiter trace sync limit must be non-negative")
    if max_non_log_bytes < 0:
        raise ValueError("Jupiter non-log size limit must be non-negative")

    destination.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    logs_dir = str(PurePosixPath(run.experiment_dir) / "logs")
    finelog_path = _finelog_path(runner, host, logs_dir, run.job_id)
    finelog_status = "unavailable"
    if finelog_path is not None:
        if status_only:
            result = _ssh(runner, host, f"tail -n 600000 -- {shlex.quote(finelog_path)}", timeout=900)
            if result.returncode == 0:
                (destination / "finelog.log").write_text(result.stdout)
                line_count = len(result.stdout.splitlines())
                finelog_status = f"current tail ({line_count:,} {'line' if line_count == 1 else 'lines'})"
            else:
                errors.append(f"finelog: {(result.stderr or result.stdout).strip()[-240:]}")
        else:
            result = _rsync(runner, host, finelog_path, destination / "finelog.log")
            if result.returncode == 0:
                finelog_status = "synced"
            else:
                errors.append(f"finelog: {(result.stderr or result.stdout).strip()[-240:]}")
    else:
        errors.append(f"finelog: no *_{run.job_id}.out file found directly under {logs_dir}")

    if status_only:
        return JupiterArtifactSyncResult(
            finelog_status,
            "not requested",
            "not requested",
            "not requested",
            0,
            tuple(errors),
        )

    slurm_logs_status = "absent"
    if _remote_exists(runner, host, logs_dir, directory=True):
        log_arguments = (f"--exclude={PurePosixPath(finelog_path).name}",) if finelog_path else ()
        result = _rsync(
            runner,
            host,
            f"{logs_dir}/",
            destination / "slurm_logs",
            extra_arguments=log_arguments,
        )
        slurm_logs_status = "synced" if result.returncode == 0 else "unavailable"
        if result.returncode:
            errors.append(f"Slurm logs: {(result.stderr or result.stdout).strip()[-240:]}")

    ray_directories = (
        str(PurePosixPath(run.experiment_dir) / "ray_logs"),
        str(PurePosixPath(run.experiment_dir) / run.job_name / "ray_logs"),
    )
    requested_ray_directories = 0
    for remote_directory in ray_directories:
        if not _remote_exists(runner, host, remote_directory, directory=True):
            continue
        requested_ray_directories += 1
        local_name = "launcher" if requested_ray_directories == 1 else "worker"
        result = _rsync(runner, host, f"{remote_directory}/", destination / "ray_logs" / local_name)
        if result.returncode:
            errors.append(f"Ray logs: {(result.stderr or result.stdout).strip()[-240:]}")

    selected_traces = _newest_trace_directories(runner, host, run.trace_root, trace_sync_limit)
    for remote_directory in selected_traces:
        errors.extend(
            f"trace {PurePosixPath(remote_directory).name}: {error}"
            for error in _sync_trace_directory(
                runner,
                host,
                remote_directory,
                destination / "trace_jobs" / PurePosixPath(remote_directory).name,
                max_non_log_bytes,
            )
        )

    trace_scope = "all" if trace_sync_limit == 0 else "newest"
    return JupiterArtifactSyncResult(
        finelog_status,
        slurm_logs_status,
        f"{requested_ray_directories} directories requested",
        f"{trace_scope} {len(selected_traces)} selected",
        len(selected_traces),
        tuple(errors),
    )
