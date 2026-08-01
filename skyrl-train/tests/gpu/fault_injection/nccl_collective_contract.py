"""Opt-in ProcessGroupNCCL collective contract tests.

Run on an otherwise idle node with at least four GPUs:

    uv run --isolated --extra dev --extra vllm \
        pytest -s tests/gpu/fault_injection/nccl_collective_contract.py

The controller launches disposable torchrun gangs for one healthy EP collective
and four expected failures. This file intentionally does not match pytest's
default ``test_*.py`` pattern.
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
from skyrl_train.utils.constants import DEFAULT_NCCL_TRACE_BUFFER_SIZE


WORLD_SIZE = 4
WARMUP_ROUNDS = 3
EP_ALL_TO_ALL_VALUES = 128
RANK_VALUE_STRIDE = 100
DIVERGENT_EP_RANKS = frozenset({0, 3})
DIVERGENT_FSDP_RANKS = frozenset(range(WORLD_SIZE)) - DIVERGENT_EP_RANKS
COLLECTIVE_TIMEOUT_SECONDS = 8
SETUP_TIMEOUT_SECONDS = 180
RUN_TIMEOUT_SECONDS = 45
REAP_TIMEOUT_SECONDS = 10
CONTROL_POLL_SECONDS = 0.1
START_SENTINEL = "start"
READY_SENTINEL_PREFIX = "ready-"
ACTIVE_SENTINEL_PREFIX = "active-"
CONTROL_DIRECTORY_ENV_VAR = "SKYRL_FAULT_CONTROL_DIR"
COMMUNICATOR_NONBLOCKING_ENVIRONMENT = {
    "TORCH_NCCL_USE_COMM_NONBLOCKING": "1",
    "TORCH_NCCL_NONBLOCKING_TIMEOUT": str(COLLECTIVE_TIMEOUT_SECONDS),
}
SKYRL_TRAIN_ROOT = Path(__file__).parents[3]
REQUIRES_FOUR_CUDA_DEVICES = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"requires {WORLD_SIZE} CUDA devices",
)


class RunMode(StrEnum):
    EP_ALL_TO_ALL = "ep-all-to-all"
    WARMED_PHASE_DIVERGENCE = "warmed-phase-divergence"
    SUBGROUP_NONARRIVAL = "subgroup-nonarrival"
    WORLD_NONARRIVAL = "world-nonarrival"
    RANK_EXIT = "rank-exit"


class CommunicatorMode(StrEnum):
    BLOCKING = "blocking"
    NONBLOCKING = "nonblocking"


FAULT_MODES = (
    RunMode.SUBGROUP_NONARRIVAL,
    RunMode.WORLD_NONARRIVAL,
    RunMode.RANK_EXIT,
)


@dataclass(frozen=True)
class RunResult:
    returncode: int
    output: str


def _wait_for_start(control_dir: Path) -> None:
    start_path = control_dir / START_SENTINEL
    while not start_path.exists():
        time.sleep(CONTROL_POLL_SECONDS)


def _hold_out(mode: RunMode, rank: int) -> None:
    print(f"FAULT_INJECTION_WITHHELD mode={mode.value} rank={rank}", flush=True)
    signal.pause()


def _wait_for_peer_activity(control_dir: Path) -> None:
    while len(tuple(control_dir.glob(f"{ACTIVE_SENTINEL_PREFIX}*"))) < WORLD_SIZE - 1:
        time.sleep(CONTROL_POLL_SECONDS)


def _run_ep_all_to_all(subgroup: dist.ProcessGroup, rank: int, device: torch.device) -> None:
    group_ranks = dist.get_process_group_ranks(subgroup)
    group_rank = group_ranks.index(rank)
    assert EP_ALL_TO_ALL_VALUES % len(group_ranks) == 0
    values_per_peer = EP_ALL_TO_ALL_VALUES // len(group_ranks)
    input_values = (
        torch.arange(
            len(group_ranks) * values_per_peer,
            device=device,
            dtype=torch.int64,
        )
        + rank * RANK_VALUE_STRIDE
    )
    output_values = torch.empty_like(input_values)
    dist.all_to_all_single(output_values, input_values, group=subgroup)
    expected_values = torch.tensor(
        [
            source_rank * RANK_VALUE_STRIDE + group_rank * values_per_peer + offset
            for source_rank in group_ranks
            for offset in range(values_per_peer)
        ],
        device=device,
        dtype=torch.int64,
    )
    torch.testing.assert_close(output_values, expected_values)


def _run_fsdp_all_gather(subgroup: dist.ProcessGroup, rank: int, device: torch.device) -> None:
    group_ranks = dist.get_process_group_ranks(subgroup)
    input_values = torch.tensor(
        [rank * RANK_VALUE_STRIDE, rank * RANK_VALUE_STRIDE + 1], device=device, dtype=torch.int64
    )
    output_values = torch.empty(len(group_ranks) * input_values.numel(), device=device, dtype=torch.int64)
    dist.all_gather_into_tensor(output_values, input_values, group=subgroup)
    expected_values = torch.tensor(
        [
            value
            for source_rank in group_ranks
            for value in (source_rank * RANK_VALUE_STRIDE, source_rank * RANK_VALUE_STRIDE + 1)
        ],
        device=device,
        dtype=torch.int64,
    )
    torch.testing.assert_close(output_values, expected_values)


def _warm_ep_and_fsdp_communicators(
    *,
    ep_group: dist.ProcessGroup,
    fsdp_group: dist.ProcessGroup,
    rank: int,
    device: torch.device,
) -> None:
    for _ in range(WARMUP_ROUNDS):
        _run_ep_all_to_all(ep_group, rank, device)
        _run_fsdp_all_gather(fsdp_group, rank, device)
    dist.barrier()
    print(f"COMMUNICATOR_WARMUP_COMPLETED rank={rank} rounds={WARMUP_ROUNDS}", flush=True)


def _worker(mode: RunMode) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    control_dir = Path(os.environ[CONTROL_DIRECTORY_ENV_VAR])

    init_worker_process_group_with_device(timeout_seconds=COLLECTIVE_TIMEOUT_SECONDS)
    device = torch.device("cuda", local_rank)
    ep_group = None
    fsdp_group = None
    if mode in (RunMode.EP_ALL_TO_ALL, RunMode.WARMED_PHASE_DIVERGENCE, RunMode.SUBGROUP_NONARRIVAL):
        mesh = create_device_mesh(WORLD_SIZE, fsdp_size=2, ep_size=2)
        ep_group = mesh["ep"].get_group()
        if mode is RunMode.WARMED_PHASE_DIVERGENCE:
            fsdp_group = mesh["fsdp"].get_group()
            _warm_ep_and_fsdp_communicators(
                ep_group=ep_group,
                fsdp_group=fsdp_group,
                rank=rank,
                device=device,
            )

    print(
        f"FAULT_INJECTION_READY mode={mode.value} rank={rank} "
        f"requested_timeout={COLLECTIVE_TIMEOUT_SECONDS} backend={dist.get_backend()}",
        flush=True,
    )
    (control_dir / f"{READY_SENTINEL_PREFIX}{rank}").touch()
    _wait_for_start(control_dir)

    if mode is RunMode.EP_ALL_TO_ALL:
        assert ep_group is not None
        _run_ep_all_to_all(ep_group, rank, device)
        print(f"EP_ALL_TO_ALL_COMPLETED rank={rank}", flush=True)
        dist.destroy_process_group()
        return
    if mode is RunMode.WARMED_PHASE_DIVERGENCE:
        assert ep_group is not None
        assert fsdp_group is not None
        if rank in DIVERGENT_EP_RANKS:
            group_ranks = dist.get_process_group_ranks(ep_group)
            assert not set(group_ranks).issubset(DIVERGENT_EP_RANKS)
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank} phase=ep-all-to-all", flush=True)
            _run_ep_all_to_all(ep_group, rank, device)
        else:
            group_ranks = dist.get_process_group_ranks(fsdp_group)
            assert not set(group_ranks).issubset(DIVERGENT_FSDP_RANKS)
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank} phase=fsdp-all-gather", flush=True)
            _run_fsdp_all_gather(fsdp_group, rank, device)
    elif mode is RunMode.SUBGROUP_NONARRIVAL:
        if rank == 0:
            assert ep_group is not None
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank}", flush=True)
            dist.all_reduce(torch.ones(1, device=device), group=ep_group)
        else:
            _hold_out(mode, rank)
    elif mode is RunMode.WORLD_NONARRIVAL:
        if rank == 0:
            _hold_out(mode, rank)
        else:
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank}", flush=True)
            dist.all_reduce(torch.ones(1, device=device))
    elif mode is RunMode.RANK_EXIT:
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


def _kill_and_reap_torchrun(process: subprocess.Popen[str]) -> bool:
    """Kill the subprocess group and report whether its leader was reaped."""
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # torchrun may finish between poll() and killpg(); wait() below
            # still reaps its leader and records the teardown result.
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

    reaped = _kill_and_reap_torchrun(process)
    pytest.fail(
        f"not all ranks completed setup within {SETUP_TIMEOUT_SECONDS}s; "
        f"process group reaped={reaped}; output:\n{_read_output(log_path)}",
        pytrace=False,
    )


def _run(mode: RunMode, *, communicator_mode: CommunicatorMode) -> RunResult:
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
    if communicator_mode is CommunicatorMode.NONBLOCKING:
        env.update(COMMUNICATOR_NONBLOCKING_ENVIRONMENT)
    else:
        for variable in COMMUNICATOR_NONBLOCKING_ENVIRONMENT:
            env.pop(variable, None)
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
                returncode = process.wait(timeout=RUN_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                reaped = _kill_and_reap_torchrun(process)
                pytest.fail(
                    f"{mode.value} did not finish within {RUN_TIMEOUT_SECONDS}s after setup; "
                    f"process group reaped={reaped}; output:\n{_read_output(log_path)}",
                    pytrace=False,
                )
        return RunResult(returncode, _read_output(log_path))


@pytest.mark.parametrize("mode", FAULT_MODES)
@REQUIRES_FOUR_CUDA_DEVICES
def test_nccl_fault_terminates_torchrun_gang(mode: RunMode) -> None:
    result = _run(mode, communicator_mode=CommunicatorMode.NONBLOCKING)

    assert f"FAULT_INJECTION_ACTIVE mode={mode.value}" in result.output
    assert "FAULT_INJECTION_UNEXPECTED_COMPLETION" not in result.output
    assert result.returncode != 0, result.output


@REQUIRES_FOUR_CUDA_DEVICES
def test_ep_all_to_all_completes_with_production_communicator_mode() -> None:
    result = _run(RunMode.EP_ALL_TO_ALL, communicator_mode=CommunicatorMode.BLOCKING)

    assert result.output.count("EP_ALL_TO_ALL_COMPLETED") == WORLD_SIZE, result.output
    assert result.returncode == 0, result.output


@REQUIRES_FOUR_CUDA_DEVICES
def test_warmed_production_phase_divergence_terminates_torchrun_gang() -> None:
    result = _run(RunMode.WARMED_PHASE_DIVERGENCE, communicator_mode=CommunicatorMode.BLOCKING)

    assert result.output.count("COMMUNICATOR_WARMUP_COMPLETED") == WORLD_SIZE, result.output
    assert result.output.count(f"FAULT_INJECTION_ACTIVE mode={RunMode.WARMED_PHASE_DIVERGENCE.value}") == WORLD_SIZE, (
        result.output
    )
    assert "FAULT_INJECTION_UNEXPECTED_COMPLETION" not in result.output
    assert result.returncode != 0, result.output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--worker",
        choices=tuple(mode.value for mode in RunMode),
        required=True,
    )
    _worker(RunMode(parser.parse_args().worker))
