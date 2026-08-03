"""Inject permanent EP/FSDP phase divergence into a warmed four-node mesh."""

from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path

import torch
import torch.distributed as dist

from tests.gpu.fault_injection.collective_payloads import (
    MeshCollectives,
    VERIFICATION_DTYPE,
    numel_for_mib,
    run_verified_all_gather,
    run_verified_all_to_all,
    warm_ep_and_fsdp_communicators,
)
from tests.gpu.fault_injection.multi_node_mesh import MeshRuntime, multi_node_mesh_runtime
from tests.process_gang import signal_rank_ready_and_wait_for_start
from tests.nccl_environment import disable_nccl_communicator_nonblocking


PROCESS_GROUP_TIMEOUT_SECONDS = 60
WARMUP_ROUNDS = 3
PAYLOAD_MIB = 1
WARMUP_MARKER = "MULTI_NODE_COMMUNICATOR_WARMUP_COMPLETED"
READY_MARKER = "MULTI_NODE_FAULT_READY"
ACTIVE_MARKER = "MULTI_NODE_FAULT_ACTIVE"
UNAFFECTED_EP_COMPLETION_MARKER = "MULTI_NODE_UNAFFECTED_EP_COMPLETED"
UNEXPECTED_COMPLETION_MARKER = "MULTI_NODE_FAULT_UNEXPECTED_COMPLETION"


def _build_and_warm_communicators(runtime: MeshRuntime, values_per_rank: int) -> MeshCollectives:
    collectives = runtime.collective_groups()
    warm_ep_and_fsdp_communicators(
        collectives,
        rounds=WARMUP_ROUNDS,
        ep_values_per_rank=values_per_rank,
        fsdp_values_per_rank=values_per_rank,
    )
    torch.cuda.synchronize(runtime.device)
    print(f"{WARMUP_MARKER} rank={runtime.placement.rank} rounds={WARMUP_ROUNDS}", flush=True)
    return collectives


def _run(runtime: MeshRuntime, control_directory: Path) -> None:
    values_per_rank = numel_for_mib(PAYLOAD_MIB, VERIFICATION_DTYPE)
    collectives = _build_and_warm_communicators(runtime, values_per_rank)
    rank = runtime.placement.rank
    print(
        f"{READY_MARKER} rank={rank} backend={dist.get_backend()} timeout={PROCESS_GROUP_TIMEOUT_SECONDS} "
        f"ep_ranks={collectives.ep.ranks} fsdp_ranks={collectives.fsdp.ranks} "
        f"async_error_handling={os.environ.get('TORCH_NCCL_ASYNC_ERROR_HANDLING')}",
        flush=True,
    )
    signal_rank_ready_and_wait_for_start(control_directory, rank)

    if rank == 0:
        print(f"{ACTIVE_MARKER} rank=0 phase=fsdp-all-gather", flush=True)
        run_verified_all_gather(collectives.fsdp, values_per_rank)
        print(f"{UNEXPECTED_COMPLETION_MARKER} rank=0 phase=fsdp-all-gather", flush=True)
        raise AssertionError("rank 0 unexpectedly completed FSDP all-gather without its peers")

    print(f"{ACTIVE_MARKER} rank={rank} phase=ep-all-to-all", flush=True)
    run_verified_all_to_all(collectives.ep, values_per_rank)
    if 0 in collectives.ep.ranks:
        print(f"{UNEXPECTED_COMPLETION_MARKER} rank={rank} phase=ep-all-to-all", flush=True)
        raise AssertionError(f"EP group {collectives.ep.ranks} unexpectedly completed without rank 0")

    # Keep unaffected ranks alive so a blocked collective, rather than an early
    # peer exit, causes the expected gang failure.
    print(f"{UNAFFECTED_EP_COMPLETION_MARKER} rank={rank}", flush=True)
    signal.pause()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-directory", type=Path, required=True)
    arguments = parser.parse_args()
    disable_nccl_communicator_nonblocking(os.environ)
    with multi_node_mesh_runtime(PROCESS_GROUP_TIMEOUT_SECONDS, os.environ) as mesh_runtime:
        _run(mesh_runtime, arguments.control_directory)
