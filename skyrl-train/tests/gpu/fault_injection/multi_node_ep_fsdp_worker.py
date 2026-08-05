"""Exercise the production EP4/FSDP4 mesh across four four-GPU nodes.

This opt-in torchrun worker checks the physical placement before it starts any
traffic: each EP group must occupy one host, and every FSDP group must span four
hosts. It then verifies payloads for FSDP all-gather, reduce-scatter, and
all-reduce traffic, both alone and interleaved with node-local EP all-to-all.
Launch one torchrun agent per node with torchrun's ``--module`` mode. See
``README.md`` in this directory for the four-node Slurm command.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist

from tests.gpu.fault_injection.collective_payloads import (
    CollectiveGroup,
    REDUCTION_DTYPE,
    VERIFICATION_DTYPE,
    numel_for_mib,
    run_verified_all_gather,
    run_verified_all_to_all,
)
from tests.gpu.fault_injection.multi_node_geometry import (
    EP_SIZE,
    EXPECTED_NODES,
    FSDP_SIZE,
    GPUS_PER_NODE,
    WORLD_SIZE,
)
from tests.gpu.fault_injection.multi_node_mesh import MeshRuntime, multi_node_mesh_runtime
from tests.nccl_environment import disable_nccl_communicator_nonblocking


PROCESS_GROUP_TIMEOUT_SECONDS = 180
PAYLOAD_MIB_SIZES = (1, 8, 32)
CONTENTION_ROUNDS = 32
ARRIVAL_SKEW_SECONDS = 2.0
SUCCESS_MARKER = "MULTI_NODE_EP_FSDP_TRAFFIC_OK"


@dataclass(frozen=True)
class PhaseTimings:
    fsdp_only_seconds: float
    alternating_seconds: float
    skewed_arrival_seconds: float


def _assert_constant(tensor: torch.Tensor, expected: float, description: str) -> None:
    minimum, maximum = torch.aminmax(tensor)
    if minimum.item() != expected or maximum.item() != expected:
        raise AssertionError(
            f"{description}: expected every value to be {expected}, got min={minimum.item()} max={maximum.item()}"
        )


def _fsdp_reduce_scatter(
    collective: CollectiveGroup,
    payload_mib: int,
) -> None:
    numel = numel_for_mib(payload_mib, REDUCTION_DTYPE)
    chunks = [
        torch.full(
            (numel,),
            float(collective.rank + destination_index),
            dtype=REDUCTION_DTYPE,
            device=collective.device,
        )
        for destination_index in range(len(collective.ranks))
    ]
    input_values = torch.cat(chunks)
    output_values = torch.empty(numel, dtype=input_values.dtype, device=collective.device)
    dist.reduce_scatter_tensor(output_values, input_values, group=collective.process_group)
    destination_index = collective.ranks.index(collective.rank)
    expected = float(sum(collective.ranks) + destination_index * len(collective.ranks))
    _assert_constant(output_values, expected, f"FSDP reduce-scatter destination rank {collective.rank}")


def _fsdp_all_reduce(
    collective: CollectiveGroup,
    payload_mib: int,
) -> None:
    numel = numel_for_mib(payload_mib, REDUCTION_DTYPE)
    values = torch.full((numel,), float(collective.rank), dtype=REDUCTION_DTYPE, device=collective.device)
    dist.all_reduce(values, group=collective.process_group)
    _assert_constant(values, float(sum(collective.ranks)), f"FSDP all-reduce rank {collective.rank}")


def _run_traffic(runtime: MeshRuntime) -> PhaseTimings:
    collectives = runtime.collective_groups()

    phase_start = time.monotonic()
    for size in PAYLOAD_MIB_SIZES:
        run_verified_all_gather(collectives.fsdp, numel_for_mib(size, VERIFICATION_DTYPE))
        _fsdp_reduce_scatter(collectives.fsdp, size)
        _fsdp_all_reduce(collectives.fsdp, size)
    torch.cuda.synchronize(runtime.device)
    fsdp_only_seconds = time.monotonic() - phase_start

    phase_start = time.monotonic()
    for round_index in range(CONTENTION_ROUNDS):
        size = PAYLOAD_MIB_SIZES[round_index % len(PAYLOAD_MIB_SIZES)]
        run_verified_all_to_all(collectives.ep, numel_for_mib(size, VERIFICATION_DTYPE))
        run_verified_all_gather(collectives.fsdp, numel_for_mib(size, VERIFICATION_DTYPE))
        run_verified_all_to_all(collectives.ep, numel_for_mib(size, VERIFICATION_DTYPE))
        _fsdp_reduce_scatter(collectives.fsdp, size)
        _fsdp_all_reduce(collectives.fsdp, size)
    torch.cuda.synchronize(runtime.device)
    alternating_seconds = time.monotonic() - phase_start

    dist.barrier()
    phase_start = time.monotonic()
    if runtime.placement.fsdp_coordinate == runtime.placement.ep_coordinate:
        # One rank in every inter-node FSDP group arrives late. The delay is the
        # input under test, not a readiness mechanism.
        time.sleep(ARRIVAL_SKEW_SECONDS)
    run_verified_all_gather(collectives.fsdp, numel_for_mib(max(PAYLOAD_MIB_SIZES), VERIFICATION_DTYPE))
    _fsdp_reduce_scatter(collectives.fsdp, max(PAYLOAD_MIB_SIZES))
    torch.cuda.synchronize(runtime.device)
    return PhaseTimings(
        fsdp_only_seconds,
        alternating_seconds,
        time.monotonic() - phase_start,
    )


def _max_timings(local_timings: PhaseTimings, device: torch.device) -> PhaseTimings:
    values = torch.tensor(
        [
            local_timings.fsdp_only_seconds,
            local_timings.alternating_seconds,
            local_timings.skewed_arrival_seconds,
        ],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return PhaseTimings(*values.cpu().tolist())


def _run_and_report_traffic(runtime: MeshRuntime) -> None:
    timings = _max_timings(
        _run_traffic(runtime),
        runtime.device,
    )
    if runtime.placement.rank == 0:
        print(
            f"{SUCCESS_MARKER} world={WORLD_SIZE} nodes={EXPECTED_NODES} gpus_per_node={GPUS_PER_NODE} "
            f"ep={EP_SIZE} fsdp={FSDP_SIZE} payload_mib={','.join(map(str, PAYLOAD_MIB_SIZES))} "
            f"rounds={CONTENTION_ROUNDS} max_phase_seconds={timings}",
            flush=True,
        )


if __name__ == "__main__":
    disable_nccl_communicator_nonblocking(os.environ)
    with multi_node_mesh_runtime(PROCESS_GROUP_TIMEOUT_SECONDS, os.environ) as mesh_runtime:
        _run_and_report_traffic(mesh_runtime)
