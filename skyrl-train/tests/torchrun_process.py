"""Bounded lifecycle helpers for disposable torchrun gangs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path


_NCCL_COMMUNICATOR_NONBLOCKING_ENVIRONMENT = {
    "TORCH_NCCL_USE_COMM_NONBLOCKING": "1",
    "TORCH_NCCL_NONBLOCKING_TIMEOUT": "",
}


def nccl_communicator_nonblocking_environment(timeout_seconds: float) -> dict[str, str]:
    """Return NCCL communicator-nonblocking settings for an explicit timeout."""

    environment = dict(_NCCL_COMMUNICATOR_NONBLOCKING_ENVIRONMENT)
    environment["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = str(timeout_seconds)
    return environment


def disable_nccl_communicator_nonblocking(environment: MutableMapping[str, str]) -> None:
    """Remove communicator-nonblocking settings from a worker environment."""

    for variable in _NCCL_COMMUNICATOR_NONBLOCKING_ENVIRONMENT:
        environment.pop(variable, None)


@dataclass(frozen=True)
class TorchrunResult:
    returncode: int
    output: str


class TorchrunTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float, reaped: bool, output: str) -> None:
        super().__init__(f"torchrun exceeded {timeout_seconds}s; process group reaped={reaped}; output:\n{output}")


@dataclass
class TorchrunGang:
    process: subprocess.Popen[str]
    directory: Path
    log_path: Path
    reap_timeout_seconds: float

    def output(self) -> str:
        """Read captured output, tolerating partial non-UTF-8 writes."""

        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""

    def kill_and_reap(self) -> bool:
        """Kill the subprocess group and report whether its leader was reaped."""

        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                # The gang may exit between poll() and killpg(); wait() still
                # reaps its leader and establishes the cleanup result.
                pass
        try:
            self.process.wait(timeout=self.reap_timeout_seconds)
        except subprocess.TimeoutExpired:
            return False
        return True

    def wait(self, timeout_seconds: float) -> TorchrunResult:
        """Wait for completion or kill and reap the gang at the deadline."""

        try:
            returncode = self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            reaped = self.kill_and_reap()
            raise TorchrunTimeoutError(timeout_seconds, reaped, self.output()) from error
        return TorchrunResult(returncode, self.output())


@contextmanager
def launch_torchrun(
    *,
    script: Path,
    arguments: Sequence[str],
    world_size: int,
    working_directory: Path,
    environment: Mapping[str, str],
    temporary_prefix: str,
    reap_timeout_seconds: float,
    control_directory_environment_variable: str | None = None,
) -> Iterator[TorchrunGang]:
    """Launch an isolated torchrun gang and guarantee bounded cleanup."""

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        str(script),
        *arguments,
    ]
    with tempfile.TemporaryDirectory(prefix=temporary_prefix) as temporary_dir:
        directory = Path(temporary_dir)
        log_path = directory / "torchrun.log"
        process_environment = dict(environment)
        if control_directory_environment_variable is not None:
            process_environment[control_directory_environment_variable] = str(directory)
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=working_directory,
                env=process_environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                text=True,
            )
            gang = TorchrunGang(process, directory, log_path, reap_timeout_seconds)
            try:
                yield gang
            finally:
                if process.poll() is None and not gang.kill_and_reap():
                    raise RuntimeError(f"failed to reap torchrun process group {process.pid}")
