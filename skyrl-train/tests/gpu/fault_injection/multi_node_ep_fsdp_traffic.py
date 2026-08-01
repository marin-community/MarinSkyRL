"""Exercise the production EP4/FSDP4 mesh across four four-GPU nodes.

This opt-in torchrun worker checks the physical placement before it starts any
traffic: each EP group must occupy one host, and every FSDP group must span four
hosts. It then verifies payloads for FSDP all-gather, reduce-scatter, and
all-reduce traffic, both alone and interleaved with node-local EP all-to-all.

Launch one torchrun agent per node with torchrun's ``--module`` mode. See
``README.md`` in this directory for a generic four-node Slurm command.
"""

from __future__ import annotations

import os
import socket
import time
from collections import Counter
from dataclasses import dataclass
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh

from skyrl_train.distributed.fsdp_utils import create_device_mesh
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from tests.gpu.fault_injection.collective_payloads import (
    CollectiveGroup,
    run_verified_all_gather,
    run_verified_all_to_all,
)
from tests.torchrun_process import disable_nccl_communicator_nonblocking


EXPECTED_NODES = 4
GPUS_PER_NODE = 4
WORLD_SIZE = EXPECTED_NODES * GPUS_PER_NODE
EP_SIZE = GPUS_PER_NODE
FSDP_SIZE = EXPECTED_NODES
PROCESS_GROUP_TIMEOUT_SECONDS = 180
PAYLOAD_MIB = (1, 8, 32)
CONTENTION_ROUNDS = 32
ARRIVAL_SKEW_SECONDS = 2.0
SUCCESS_MARKER = "MULTI_NODE_EP_FSDP_TRAFFIC_OK"


@dataclass(frozen=True)
class MeshPlacement:
    rank: int
    ep_coordinate: int
    fsdp_coordinate: int


@dataclass(frozen=True)
class PhaseTimings:
    fsdp_only_seconds: float
    alternating_seconds: float
    skewed_arrival_seconds: float


def _validate_mesh_placement(mesh: DeviceMesh) -> MeshPlacement:
    rank = dist.get_rank()
    local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])
    hostname = socket.gethostname()

    if dist.get_world_size() != WORLD_SIZE:
        raise AssertionError(f"expected {WORLD_SIZE} ranks, got {dist.get_world_size()}")
    if local_world_size != GPUS_PER_NODE:
        raise AssertionError(f"expected {GPUS_PER_NODE} ranks per node, got {local_world_size}")

    hostnames: list[str | None] = [None] * WORLD_SIZE
    dist.all_gather_object(hostnames, hostname)
    if any(item is None for item in hostnames):
        raise AssertionError(f"missing hostnames from rank inventory: {hostnames}")
    resolved_hostnames = tuple(str(item) for item in hostnames)
    host_counts = Counter(resolved_hostnames)
    if len(host_counts) != EXPECTED_NODES or set(host_counts.values()) != {GPUS_PER_NODE}:
        raise AssertionError(
            f"expected {EXPECTED_NODES} hosts with {GPUS_PER_NODE} ranks each, got {dict(host_counts)}"
        )

    ep_ranks = tuple(dist.get_process_group_ranks(mesh["ep"].get_group()))
    fsdp_ranks = tuple(dist.get_process_group_ranks(mesh["fsdp"].get_group()))
    coordinates = dict(zip(mesh.mesh_dim_names, mesh.get_coordinate(), strict=True))
    ep_hosts = {resolved_hostnames[group_rank] for group_rank in ep_ranks}
    fsdp_hosts = {resolved_hostnames[group_rank] for group_rank in fsdp_ranks}
    if ep_hosts != {hostname}:
        raise AssertionError(f"EP group {ep_ranks} crosses hosts: {sorted(ep_hosts)}")
    if len(fsdp_hosts) != EXPECTED_NODES:
        raise AssertionError(f"FSDP group {fsdp_ranks} spans {len(fsdp_hosts)} hosts: {sorted(fsdp_hosts)}")

    return MeshPlacement(
        rank,
        coordinates["ep"],
        coordinates["fsdp"],
    )


def _numel_for_mib(payload_mib: int, dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return payload_mib * 1024 * 1024 // element_size


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
    numel = _numel_for_mib(payload_mib, torch.float32)
    chunks = [
        torch.full((numel,), float(collective.rank + destination_index), device=collective.device)
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
    numel = _numel_for_mib(payload_mib, torch.float32)
    values = torch.full((numel,), float(collective.rank), device=collective.device)
    dist.all_reduce(values, group=collective.process_group)
    _assert_constant(values, float(sum(collective.ranks)), f"FSDP all-reduce rank {collective.rank}")


def _run_traffic(
    mesh: DeviceMesh,
    placement: MeshPlacement,
    device: torch.device,
) -> PhaseTimings:
    ep = CollectiveGroup.from_process_group(mesh["ep"].get_group(), placement.rank, device)
    fsdp = CollectiveGroup.from_process_group(mesh["fsdp"].get_group(), placement.rank, device)

    phase_start = time.monotonic()
    for size in PAYLOAD_MIB:
        run_verified_all_gather(fsdp, _numel_for_mib(size, torch.int64))
        _fsdp_reduce_scatter(fsdp, size)
        _fsdp_all_reduce(fsdp, size)
    torch.cuda.synchronize(device)
    fsdp_only_seconds = time.monotonic() - phase_start

    phase_start = time.monotonic()
    for round_index in range(CONTENTION_ROUNDS):
        size = PAYLOAD_MIB[round_index % len(PAYLOAD_MIB)]
        run_verified_all_to_all(ep, _numel_for_mib(size, torch.int64))
        run_verified_all_gather(fsdp, _numel_for_mib(size, torch.int64))
        run_verified_all_to_all(ep, _numel_for_mib(size, torch.int64))
        _fsdp_reduce_scatter(fsdp, size)
        _fsdp_all_reduce(fsdp, size)
    torch.cuda.synchronize(device)
    alternating_seconds = time.monotonic() - phase_start

    dist.barrier()
    phase_start = time.monotonic()
    if placement.fsdp_coordinate == placement.ep_coordinate:
        # One rank in every inter-node FSDP group arrives late. The delay is the
        # input under test, not a readiness mechanism.
        time.sleep(ARRIVAL_SKEW_SECONDS)
    run_verified_all_gather(fsdp, _numel_for_mib(max(PAYLOAD_MIB), torch.int64))
    _fsdp_reduce_scatter(fsdp, max(PAYLOAD_MIB))
    torch.cuda.synchronize(device)
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


def _run_with_process_group() -> None:
    init_worker_process_group_with_device(timeout_seconds=PROCESS_GROUP_TIMEOUT_SECONDS)
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device("cuda", local_rank)
        mesh = create_device_mesh(WORLD_SIZE, fsdp_size=FSDP_SIZE, ep_size=EP_SIZE)
        placement = _validate_mesh_placement(mesh)
        dist.barrier()
        timings = _max_timings(
            _run_traffic(mesh, placement, device),
            device,
        )
        if placement.rank == 0:
            print(
                f"{SUCCESS_MARKER} world={WORLD_SIZE} nodes={EXPECTED_NODES} gpus_per_node={GPUS_PER_NODE} "
                f"ep={EP_SIZE} fsdp={FSDP_SIZE} payload_mib={','.join(map(str, PAYLOAD_MIB))} "
                f"rounds={CONTENTION_ROUNDS} max_phase_seconds={timings}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run() -> None:
    disable_nccl_communicator_nonblocking(os.environ)
    # Device-mesh subgroups are created after WORLD initialization and consume
    # the production timeout environment variable, not the WORLD timeout arg.
    with patch.dict(
        os.environ,
        {"SKYRL_WORKER_NCCL_TIMEOUT_IN_S": str(PROCESS_GROUP_TIMEOUT_SECONDS)},
    ):
        _run_with_process_group()


if __name__ == "__main__":
    _run()
