"""Opt-in ProcessGroupNCCL collective contract tests.

Run on an otherwise idle node with at least four GPUs:

    uv run --isolated --group dev --extra vllm \
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
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from skyrl_train.distributed.fsdp_utils import create_device_mesh
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from skyrl_train.utils.nccl_environment import nccl_diagnostics_environment
from tests.gpu.fault_injection.collective_payloads import (
    MeshCollectives,
    run_verified_all_gather,
    run_verified_all_to_all,
    warm_ep_and_fsdp_communicators,
)
from tests.gpu.fault_injection.fault_injection_paths import SKYRL_TRAIN_ROOT
from tests.gpu.fault_injection.single_node_runtime import (
    REAP_TIMEOUT_SECONDS,
    REQUIRES_FOUR_CUDA_DEVICES,
    WORLD_SIZE,
)
from tests.nccl_environment import (
    disable_nccl_communicator_nonblocking,
    nccl_communicator_nonblocking_environment,
)
from tests.process_gang import (
    CONTROL_POLL_SECONDS,
    ProcessGangResult,
    launch_torchrun,
    run_after_rank_readiness,
    signal_rank_ready_and_wait_for_start,
)


WARMUP_ROUNDS = 3
EP_ALL_TO_ALL_VALUES = 128
FSDP_ALL_GATHER_VALUES = 2
DIVERGENT_EP_RANKS = frozenset({0, 3})
DIVERGENT_FSDP_RANKS = frozenset(range(WORLD_SIZE)) - DIVERGENT_EP_RANKS
COLLECTIVE_TIMEOUT_SECONDS = 8
# Detect a stalled watchdog thread inside the collective deadline. This is not
# a collective-progress timeout.
HEARTBEAT_TIMEOUT_SECONDS = 5
SETUP_TIMEOUT_SECONDS = 180
RUN_TIMEOUT_SECONDS = 45
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


def _hold_out(mode: RunMode, rank: int) -> None:
    print(f"FAULT_INJECTION_WITHHELD mode={mode.value} rank={rank}", flush=True)
    signal.pause()


def _wait_for_peer_activity(control_dir: Path) -> None:
    while len(tuple(control_dir.glob(f"{ACTIVE_SENTINEL_PREFIX}*"))) < WORLD_SIZE - 1:
        time.sleep(CONTROL_POLL_SECONDS)


def _mesh_collectives(rank: int, device: torch.device) -> MeshCollectives:
    mesh = create_device_mesh(
        WORLD_SIZE,
        fsdp_size=2,
        ep_size=2,
        timeout_seconds=COLLECTIVE_TIMEOUT_SECONDS,
    )
    return MeshCollectives.from_mesh(mesh, rank, device)


def _await_fault_start(mode: RunMode, rank: int, control_directory: Path) -> None:
    print(
        f"FAULT_INJECTION_READY mode={mode.value} rank={rank} "
        f"requested_timeout={COLLECTIVE_TIMEOUT_SECONDS} backend={dist.get_backend()}",
        flush=True,
    )
    signal_rank_ready_and_wait_for_start(control_directory, rank)


def _report_unexpected_completion(mode: RunMode, rank: int) -> None:
    print(f"FAULT_INJECTION_UNEXPECTED_COMPLETION mode={mode.value} rank={rank}", flush=True)


def _run_ep_all_to_all(rank: int, device: torch.device, control_directory: Path) -> None:
    ep = _mesh_collectives(rank, device).ep
    _await_fault_start(RunMode.EP_ALL_TO_ALL, rank, control_directory)
    run_verified_all_to_all(ep, EP_ALL_TO_ALL_VALUES)
    print(f"EP_ALL_TO_ALL_COMPLETED rank={rank}", flush=True)


def _run_warmed_phase_divergence(rank: int, device: torch.device, control_directory: Path) -> None:
    collectives = _mesh_collectives(rank, device)
    warm_ep_and_fsdp_communicators(
        collectives,
        rounds=WARMUP_ROUNDS,
        ep_values_per_rank=EP_ALL_TO_ALL_VALUES,
        fsdp_values_per_rank=FSDP_ALL_GATHER_VALUES,
    )
    print(f"COMMUNICATOR_WARMUP_COMPLETED rank={rank} rounds={WARMUP_ROUNDS}", flush=True)
    _await_fault_start(RunMode.WARMED_PHASE_DIVERGENCE, rank, control_directory)

    if rank in DIVERGENT_EP_RANKS:
        assert not set(collectives.ep.ranks).issubset(DIVERGENT_EP_RANKS)
        print(
            f"FAULT_INJECTION_ACTIVE mode={RunMode.WARMED_PHASE_DIVERGENCE.value} rank={rank} phase=ep-all-to-all",
            flush=True,
        )
        run_verified_all_to_all(collectives.ep, EP_ALL_TO_ALL_VALUES)
    else:
        assert not set(collectives.fsdp.ranks).issubset(DIVERGENT_FSDP_RANKS)
        print(
            f"FAULT_INJECTION_ACTIVE mode={RunMode.WARMED_PHASE_DIVERGENCE.value} rank={rank} phase=fsdp-all-gather",
            flush=True,
        )
        run_verified_all_gather(collectives.fsdp, input_values_per_rank=FSDP_ALL_GATHER_VALUES)
    _report_unexpected_completion(RunMode.WARMED_PHASE_DIVERGENCE, rank)


def _run_subgroup_nonarrival(rank: int, device: torch.device, control_directory: Path) -> None:
    ep = _mesh_collectives(rank, device).ep
    _await_fault_start(RunMode.SUBGROUP_NONARRIVAL, rank, control_directory)
    if rank == 0:
        print(f"FAULT_INJECTION_ACTIVE mode={RunMode.SUBGROUP_NONARRIVAL.value} rank={rank}", flush=True)
        dist.all_reduce(torch.ones(1, device=device), group=ep.process_group)
    else:
        _hold_out(RunMode.SUBGROUP_NONARRIVAL, rank)
    _report_unexpected_completion(RunMode.SUBGROUP_NONARRIVAL, rank)


def _run_world_nonarrival(rank: int, device: torch.device, control_directory: Path) -> None:
    _await_fault_start(RunMode.WORLD_NONARRIVAL, rank, control_directory)
    if rank == 0:
        _hold_out(RunMode.WORLD_NONARRIVAL, rank)
    else:
        print(f"FAULT_INJECTION_ACTIVE mode={RunMode.WORLD_NONARRIVAL.value} rank={rank}", flush=True)
        dist.all_reduce(torch.ones(1, device=device))
    _report_unexpected_completion(RunMode.WORLD_NONARRIVAL, rank)


def _run_rank_exit(rank: int, device: torch.device, control_directory: Path) -> None:
    _await_fault_start(RunMode.RANK_EXIT, rank, control_directory)
    if rank == 0:
        print(f"FAULT_INJECTION_EXIT mode={RunMode.RANK_EXIT.value} rank={rank}", flush=True)
        _wait_for_peer_activity(control_directory)
        os._exit(17)
    print(f"FAULT_INJECTION_ACTIVE mode={RunMode.RANK_EXIT.value} rank={rank}", flush=True)
    work = dist.all_reduce(torch.ones(1, device=device), async_op=True)
    (control_directory / f"{ACTIVE_SENTINEL_PREFIX}{rank}").touch()
    work.wait()
    _report_unexpected_completion(RunMode.RANK_EXIT, rank)


WORKERS_BY_MODE: dict[RunMode, Callable[[int, torch.device, Path], None]] = {
    RunMode.EP_ALL_TO_ALL: _run_ep_all_to_all,
    RunMode.WARMED_PHASE_DIVERGENCE: _run_warmed_phase_divergence,
    RunMode.SUBGROUP_NONARRIVAL: _run_subgroup_nonarrival,
    RunMode.WORLD_NONARRIVAL: _run_world_nonarrival,
    RunMode.RANK_EXIT: _run_rank_exit,
}


def _worker(mode: RunMode) -> None:
    rank = int(os.environ["RANK"])
    control_directory = Path(os.environ[CONTROL_DIRECTORY_ENV_VAR])
    init_worker_process_group_with_device(timeout_seconds=COLLECTIVE_TIMEOUT_SECONDS)
    try:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        WORKERS_BY_MODE[mode](rank, device, control_directory)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(mode: RunMode, *, communicator_mode: CommunicatorMode) -> ProcessGangResult:
    env = os.environ.copy()
    env.update(
        nccl_diagnostics_environment(
            heartbeat_timeout_seconds=HEARTBEAT_TIMEOUT_SECONDS,
        )
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
        return run_after_rank_readiness(
            gang,
            expected_ranks=WORLD_SIZE,
            setup_timeout_seconds=SETUP_TIMEOUT_SECONDS,
            run_timeout_seconds=RUN_TIMEOUT_SECONDS,
        )


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
