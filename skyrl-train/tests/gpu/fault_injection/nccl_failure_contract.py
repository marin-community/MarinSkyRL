"""Opt-in ProcessGroupNCCL failure-contract tests.

Run on an otherwise idle node with at least four GPUs:

    uv run --isolated --extra dev --extra vllm \
        pytest -s tests/gpu/fault_injection/nccl_failure_contract.py

The controller launches disposable torchrun gangs that are expected to fail.
This file intentionally does not match pytest's default ``test_*.py`` pattern.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from skyrl_train.distributed.fsdp_utils import create_device_mesh
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from skyrl_train.utils.constants import DEFAULT_NCCL_TRACE_BUFFER_SIZE, nccl_communicator_timeout_environment


WORLD_SIZE = 4
COLLECTIVE_TIMEOUT_SECONDS = 8
SETUP_TIMEOUT_SECONDS = 180
FAULT_TIMEOUT_SECONDS = 45
REAP_TIMEOUT_SECONDS = 10
CONTROL_POLL_SECONDS = 0.1
START_SENTINEL = "start"
READY_SENTINEL_PREFIX = "ready-"
ACTIVE_SENTINEL_PREFIX = "active-"
CONTROL_DIRECTORY_ENV_VAR = "SKYRL_FAULT_CONTROL_DIR"
SKYRL_TRAIN_ROOT = Path(__file__).parents[3]


class FaultMode(StrEnum):
    SUBGROUP_NONARRIVAL = "subgroup-nonarrival"
    WORLD_NONARRIVAL = "world-nonarrival"
    RANK_EXIT = "rank-exit"


FAULT_MODES = tuple(FaultMode)


@dataclass(frozen=True)
class FaultRun:
    returncode: int
    output: str


def _wait_for_start(control_dir: Path) -> None:
    start_path = control_dir / START_SENTINEL
    while not start_path.exists():
        time.sleep(CONTROL_POLL_SECONDS)


def _hold_out(mode: FaultMode, rank: int) -> None:
    print(f"FAULT_INJECTION_WITHHELD mode={mode.value} rank={rank}", flush=True)
    time.sleep(FAULT_TIMEOUT_SECONDS * 2)


def _wait_for_peer_activity(control_dir: Path) -> None:
    while len(tuple(control_dir.glob(f"{ACTIVE_SENTINEL_PREFIX}*"))) < WORLD_SIZE - 1:
        time.sleep(CONTROL_POLL_SECONDS)


def _worker(mode: FaultMode) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    control_dir = Path(os.environ[CONTROL_DIRECTORY_ENV_VAR])

    init_worker_process_group_with_device(timeout_seconds=COLLECTIVE_TIMEOUT_SECONDS)
    device = torch.device("cuda", local_rank)
    subgroup = None
    if mode is FaultMode.SUBGROUP_NONARRIVAL:
        mesh = create_device_mesh(WORLD_SIZE, fsdp_size=2, ep_size=2)
        subgroup = mesh["ep"].get_group()

    print(
        f"FAULT_INJECTION_READY mode={mode.value} rank={rank} "
        f"requested_timeout={COLLECTIVE_TIMEOUT_SECONDS} backend={dist.get_backend()}",
        flush=True,
    )
    (control_dir / f"{READY_SENTINEL_PREFIX}{rank}").touch()
    _wait_for_start(control_dir)

    if mode is FaultMode.SUBGROUP_NONARRIVAL:
        if rank == 0:
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank}", flush=True)
            dist.all_reduce(torch.ones(1, device=device), group=subgroup)
        else:
            _hold_out(mode, rank)
    elif mode is FaultMode.WORLD_NONARRIVAL:
        if rank == 0:
            _hold_out(mode, rank)
        else:
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank}", flush=True)
            dist.all_reduce(torch.ones(1, device=device))
    elif mode is FaultMode.RANK_EXIT:
        if rank == 0:
            print(f"FAULT_INJECTION_EXIT mode={mode.value} rank={rank}", flush=True)
            _wait_for_peer_activity(control_dir)
            os._exit(17)
        print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank}", flush=True)
        work = dist.all_reduce(torch.ones(1, device=device), async_op=True)
        (control_dir / f"{ACTIVE_SENTINEL_PREFIX}{rank}").touch()
        work.wait()
    else:  # pragma: no cover - argparse constrains worker invocations
        raise ValueError(f"unknown fault mode: {mode}")

    print(f"FAULT_INJECTION_UNEXPECTED_COMPLETION mode={mode.value} rank={rank}", flush=True)
    dist.destroy_process_group()


def _terminate_process_group(process: subprocess.Popen[str]) -> bool:
    """Kill the subprocess group and report whether its leader was reaped."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=REAP_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return False
    return True


def _read_output(log_path: Path) -> str:
    return log_path.read_text(errors="replace") if log_path.exists() else ""


def _wait_for_all_ranks_ready(process: subprocess.Popen[str], control_dir: Path, log_path: Path) -> None:
    deadline = time.monotonic() + SETUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        ready_ranks = {
            path.name.removeprefix(READY_SENTINEL_PREFIX) for path in control_dir.glob(f"{READY_SENTINEL_PREFIX}*")
        }
        if len(ready_ranks) == WORLD_SIZE:
            return
        if process.poll() is not None:
            pytest.fail(
                f"torchrun exited during setup with code {process.returncode}; output:\n{_read_output(log_path)}",
                pytrace=False,
            )
        time.sleep(CONTROL_POLL_SECONDS)

    reaped = _terminate_process_group(process)
    pytest.fail(
        f"not all ranks completed setup within {SETUP_TIMEOUT_SECONDS}s; "
        f"process group reaped={reaped}; output:\n{_read_output(log_path)}",
        pytrace=False,
    )


def _run_fault(mode: FaultMode) -> FaultRun:
    env = os.environ.copy()
    env.update(
        {
            "SKYRL_WORKER_NCCL_TIMEOUT_IN_S": str(COLLECTIVE_TIMEOUT_SECONDS),
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_ENABLE_MONITORING": "1",
            "TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC": "5",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_FR_BUFFER_SIZE": str(DEFAULT_NCCL_TRACE_BUFFER_SIZE),
            "TORCH_NCCL_TRACE_BUFFER_SIZE": str(DEFAULT_NCCL_TRACE_BUFFER_SIZE),
        }
    )
    env.update(nccl_communicator_timeout_environment(COLLECTIVE_TIMEOUT_SECONDS))
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        str(Path(__file__).resolve()),
        "--worker",
        mode.value,
    ]
    with tempfile.TemporaryDirectory(prefix=f"skyrl-nccl-{mode.value}-") as temporary_dir:
        control_dir = Path(temporary_dir)
        log_path = control_dir / "torchrun.log"
        env[CONTROL_DIRECTORY_ENV_VAR] = str(control_dir)
        with log_path.open("w") as log_file:
            process = subprocess.Popen(
                command,
                cwd=SKYRL_TRAIN_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _wait_for_all_ranks_ready(process, control_dir, log_path)
            (control_dir / START_SENTINEL).touch()
            try:
                returncode = process.wait(timeout=FAULT_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                reaped = _terminate_process_group(process)
                pytest.fail(
                    f"{mode.value} did not tear down within {FAULT_TIMEOUT_SECONDS}s after setup; "
                    f"process group reaped={reaped}; output:\n{_read_output(log_path)}",
                    pytrace=False,
                )
        return FaultRun(returncode, _read_output(log_path))


@pytest.mark.parametrize("mode", FAULT_MODES)
def test_nccl_fault_terminates_torchrun_gang(mode: FaultMode) -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"requires {WORLD_SIZE} CUDA devices")

    result = _run_fault(mode)

    assert f"FAULT_INJECTION_ACTIVE mode={mode.value}" in result.output
    assert "FAULT_INJECTION_UNEXPECTED_COMPLETION" not in result.output
    assert result.returncode != 0, result.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        choices=tuple(mode.value for mode in FAULT_MODES),
        required=True,
    )
    _worker(FaultMode(parser.parse_args().worker))
