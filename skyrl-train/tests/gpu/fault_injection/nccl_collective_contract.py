"""Opt-in ProcessGroupNCCL collective contract tests.

Run on an otherwise idle node with at least four GPUs:

    uv run --isolated --group dev --extra train-vllm \
        pytest -s tests/gpu/fault_injection/nccl_collective_contract.py

The controller launches disposable torchrun gangs for one healthy EP collective
and four expected failures. This file intentionally does not match pytest's
default ``test_*.py`` pattern.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
from enum import StrEnum
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from skyrl_train.distributed.fsdp_utils import create_device_mesh
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from skyrl_train.utils.constants import DEFAULT_NCCL_TRACE_BUFFER_SIZE
from tests.gpu.fault_injection.collective_payloads import (
    CollectiveGroup,
    run_verified_all_gather,
    run_verified_all_to_all,
)
from tests.gpu.fault_injection.single_node_runtime import (
    REAP_TIMEOUT_SECONDS,
    REQUIRES_FOUR_CUDA_DEVICES,
    SKYRL_TRAIN_ROOT,
    WORLD_SIZE,
)
from tests.torchrun_process import (
    TorchrunGang,
    TorchrunResult,
    TorchrunTimeoutError,
    disable_nccl_communicator_nonblocking,
    launch_torchrun,
    nccl_communicator_nonblocking_environment,
)


WARMUP_ROUNDS = 3
EP_ALL_TO_ALL_VALUES = 128
FSDP_ALL_GATHER_VALUES = 2
DIVERGENT_EP_RANKS = frozenset({0, 3})
DIVERGENT_FSDP_RANKS = frozenset(range(WORLD_SIZE)) - DIVERGENT_EP_RANKS
COLLECTIVE_TIMEOUT_SECONDS = 8
SETUP_TIMEOUT_SECONDS = 180
RUN_TIMEOUT_SECONDS = 45
CONTROL_POLL_SECONDS = 0.1
START_SENTINEL = "start"
READY_SENTINEL_PREFIX = "ready-"
ACTIVE_SENTINEL_PREFIX = "active-"
CONTROL_DIRECTORY_ENV_VAR = "SKYRL_FAULT_CONTROL_DIR"
COMMUNICATOR_NONBLOCKING_ENVIRONMENT = nccl_communicator_nonblocking_environment(COLLECTIVE_TIMEOUT_SECONDS)


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


def _warm_ep_and_fsdp_communicators(
    *,
    ep: CollectiveGroup,
    fsdp: CollectiveGroup,
) -> None:
    for _ in range(WARMUP_ROUNDS):
        run_verified_all_to_all(ep, EP_ALL_TO_ALL_VALUES)
        run_verified_all_gather(fsdp, input_values_per_rank=FSDP_ALL_GATHER_VALUES)
    dist.barrier()
    print(f"COMMUNICATOR_WARMUP_COMPLETED rank={ep.rank} rounds={WARMUP_ROUNDS}", flush=True)


def _worker(mode: RunMode) -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    control_dir = Path(os.environ[CONTROL_DIRECTORY_ENV_VAR])

    init_worker_process_group_with_device(timeout_seconds=COLLECTIVE_TIMEOUT_SECONDS)
    device = torch.device("cuda", local_rank)
    ep = None
    fsdp = None
    if mode in (RunMode.EP_ALL_TO_ALL, RunMode.WARMED_PHASE_DIVERGENCE, RunMode.SUBGROUP_NONARRIVAL):
        mesh = create_device_mesh(WORLD_SIZE, fsdp_size=2, ep_size=2)
        ep = CollectiveGroup.from_process_group(mesh["ep"].get_group(), rank, device)
        if mode is RunMode.WARMED_PHASE_DIVERGENCE:
            fsdp = CollectiveGroup.from_process_group(mesh["fsdp"].get_group(), rank, device)
            _warm_ep_and_fsdp_communicators(
                ep=ep,
                fsdp=fsdp,
            )

    print(
        f"FAULT_INJECTION_READY mode={mode.value} rank={rank} "
        f"requested_timeout={COLLECTIVE_TIMEOUT_SECONDS} backend={dist.get_backend()}",
        flush=True,
    )
    (control_dir / f"{READY_SENTINEL_PREFIX}{rank}").touch()
    _wait_for_start(control_dir)

    if mode is RunMode.EP_ALL_TO_ALL:
        assert ep is not None
        run_verified_all_to_all(ep, EP_ALL_TO_ALL_VALUES)
        print(f"EP_ALL_TO_ALL_COMPLETED rank={rank}", flush=True)
        dist.destroy_process_group()
        return
    if mode is RunMode.WARMED_PHASE_DIVERGENCE:
        assert ep is not None
        assert fsdp is not None
        if rank in DIVERGENT_EP_RANKS:
            assert not set(ep.ranks).issubset(DIVERGENT_EP_RANKS)
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank} phase=ep-all-to-all", flush=True)
            run_verified_all_to_all(ep, EP_ALL_TO_ALL_VALUES)
        else:
            assert not set(fsdp.ranks).issubset(DIVERGENT_FSDP_RANKS)
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank} phase=fsdp-all-gather", flush=True)
            run_verified_all_gather(fsdp, input_values_per_rank=FSDP_ALL_GATHER_VALUES)
    elif mode is RunMode.SUBGROUP_NONARRIVAL:
        if rank == 0:
            assert ep is not None
            print(f"FAULT_INJECTION_ACTIVE mode={mode.value} rank={rank}", flush=True)
            dist.all_reduce(torch.ones(1, device=device), group=ep.process_group)
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


def _wait_for_all_ranks_ready(gang: TorchrunGang) -> None:
    deadline = time.monotonic() + SETUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        ready_ranks = {
            path.name.removeprefix(READY_SENTINEL_PREFIX) for path in gang.directory.glob(f"{READY_SENTINEL_PREFIX}*")
        }
        if len(ready_ranks) == WORLD_SIZE:
            return
        if gang.process.poll() is not None:
            pytest.fail(
                f"torchrun exited during setup with code {gang.process.returncode}; output:\n{gang.output()}",
                pytrace=False,
            )
        time.sleep(CONTROL_POLL_SECONDS)

    reaped = gang.kill_and_reap()
    pytest.fail(
        f"not all ranks completed setup within {SETUP_TIMEOUT_SECONDS}s; "
        f"process group reaped={reaped}; output:\n{gang.output()}",
        pytrace=False,
    )


def _run(mode: RunMode, *, communicator_mode: CommunicatorMode) -> TorchrunResult:
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
        disable_nccl_communicator_nonblocking(env)
    with launch_torchrun(
        script=Path(__file__).resolve(),
        arguments=("--worker", mode.value),
        world_size=WORLD_SIZE,
        working_directory=SKYRL_TRAIN_ROOT,
        environment=env,
        temporary_prefix=f"skyrl-nccl-{mode.value}-",
        reap_timeout_seconds=REAP_TIMEOUT_SECONDS,
        control_directory_environment_variable=CONTROL_DIRECTORY_ENV_VAR,
    ) as gang:
        _wait_for_all_ranks_ready(gang)
        (gang.directory / START_SENTINEL).touch()
        try:
            return gang.wait(RUN_TIMEOUT_SECONDS)
        except TorchrunTimeoutError as error:
            pytest.fail(f"{mode.value} did not finish after setup: {error}", pytrace=False)


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
