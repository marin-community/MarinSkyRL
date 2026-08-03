"""Synchronize, control, and bound the lifecycle of disposable process gangs."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


CONTROLLER_START_SENTINEL = "start"
RANK_READY_SENTINEL_PREFIX = "ready-"
CONTROL_POLL_SECONDS = 0.1


class ReapOutcome(StrEnum):
    REQUESTED_SIGNAL = "reaped-after-requested-signal"
    SIGKILL_ESCALATION = "reaped-after-sigkill-escalation"
    NOT_REAPED = "not-reaped"


class SignalWaitOutcome(StrEnum):
    EXITED = "exited"
    TIMED_OUT = "timed-out"


@dataclass(frozen=True)
class ProcessGangResult:
    returncode: int
    output: str


class ProcessGangTimeoutError(TimeoutError):
    def __init__(self, timeout_seconds: float, reap_outcome: ReapOutcome, output: str) -> None:
        super().__init__(f"process gang exceeded {timeout_seconds}s; cleanup={reap_outcome.value}; output:\n{output}")


class ProcessGangSetupError(RuntimeError):
    pass


@dataclass
class ProcessGang:
    process: subprocess.Popen[str]
    control_directory: Path
    log_path: Path
    reap_timeout_seconds: float
    termination_signal: signal.Signals = signal.SIGKILL

    def output(self) -> str:
        """Read captured output, tolerating partial non-UTF-8 writes."""

        return self.log_path.read_text(errors="replace")

    def _signal_and_wait(self, requested_signal: signal.Signals) -> SignalWaitOutcome:
        """Signal the gang and classify its leader's bounded wait."""

        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, requested_signal)
            except ProcessLookupError:
                # The gang may exit between poll() and killpg(); wait() still
                # reaps its leader and establishes the cleanup result.
                pass
        try:
            self.process.wait(timeout=self.reap_timeout_seconds)
        except subprocess.TimeoutExpired:
            return SignalWaitOutcome.TIMED_OUT
        return SignalWaitOutcome.EXITED

    def kill_and_reap(self) -> ReapOutcome:
        """Terminate the subprocess group and report how its leader was reaped."""

        if self._signal_and_wait(self.termination_signal) is SignalWaitOutcome.EXITED:
            return ReapOutcome.REQUESTED_SIGNAL
        if self.termination_signal == signal.SIGKILL:
            return ReapOutcome.NOT_REAPED
        if self._signal_and_wait(signal.SIGKILL) is SignalWaitOutcome.EXITED:
            return ReapOutcome.SIGKILL_ESCALATION
        return ReapOutcome.NOT_REAPED

    def wait(self, timeout_seconds: float) -> ProcessGangResult:
        """Wait for completion or kill and reap the gang at the deadline."""

        try:
            returncode = self.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            reap_outcome = self.kill_and_reap()
            raise ProcessGangTimeoutError(timeout_seconds, reap_outcome, self.output()) from error
        return ProcessGangResult(returncode, self.output())


def _wait_for_control_file(path: Path) -> None:
    """Wait until the controller creates a shared sentinel file."""

    while not path.exists():
        time.sleep(CONTROL_POLL_SECONDS)


def signal_rank_ready_and_wait_for_start(control_directory: Path, rank: int) -> None:
    """Signal worker readiness and wait for the controller to release the fault phase."""

    (control_directory / f"{RANK_READY_SENTINEL_PREFIX}{rank}").touch()
    _wait_for_control_file(control_directory / CONTROLLER_START_SENTINEL)


def wait_for_rank_readiness(
    gang: ProcessGang,
    *,
    expected_ranks: int,
    timeout_seconds: float,
) -> None:
    """Wait for rank readiness files or raise with bounded process cleanup."""

    deadline = time.monotonic() + timeout_seconds
    ready_paths: tuple[Path, ...] = ()
    while time.monotonic() < deadline:
        ready_paths = tuple(gang.control_directory.glob(f"{RANK_READY_SENTINEL_PREFIX}*"))
        if len(ready_paths) == expected_ranks:
            return
        if gang.process.poll() is not None:
            raise ProcessGangSetupError(
                f"process gang exited during setup with code {gang.process.returncode}; output:\n{gang.output()}"
            )
        time.sleep(CONTROL_POLL_SECONDS)

    reap_outcome = gang.kill_and_reap()
    raise ProcessGangSetupError(
        f"only {len(ready_paths)}/{expected_ranks} ranks completed setup within {timeout_seconds}s; "
        f"cleanup={reap_outcome.value}; output:\n{gang.output()}"
    )


def run_after_rank_readiness(
    gang: ProcessGang,
    *,
    expected_ranks: int,
    setup_timeout_seconds: float,
    run_timeout_seconds: float,
) -> ProcessGangResult:
    """Release a ready gang into its test phase, then wait under a separate deadline."""

    wait_for_rank_readiness(
        gang,
        expected_ranks=expected_ranks,
        timeout_seconds=setup_timeout_seconds,
    )
    (gang.control_directory / CONTROLLER_START_SENTINEL).touch()
    return gang.wait(run_timeout_seconds)


@contextmanager
def launch_process_gang(
    *,
    command: Sequence[str],
    working_directory: Path,
    environment: Mapping[str, str],
    control_directory: Path,
    log_path: Path,
    reap_timeout_seconds: float,
    process_description: str,
    termination_signal: signal.Signals = signal.SIGKILL,
) -> Iterator[ProcessGang]:
    """Launch a process gang and guarantee bounded cleanup of its session."""

    with log_path.open("w") as log_file:
        process = subprocess.Popen(
            command,
            cwd=working_directory,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )
        gang = ProcessGang(process, control_directory, log_path, reap_timeout_seconds, termination_signal)
        try:
            yield gang
        finally:
            if process.poll() is None and gang.kill_and_reap() is ReapOutcome.NOT_REAPED:
                raise RuntimeError(f"failed to reap {process_description} {process.pid}")


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
) -> Iterator[ProcessGang]:
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
        with launch_process_gang(
            command=command,
            working_directory=working_directory,
            environment=process_environment,
            control_directory=directory,
            log_path=log_path,
            reap_timeout_seconds=reap_timeout_seconds,
            process_description="torchrun process group",
        ) as gang:
            yield gang
