"""Exercise the production EP4/FSDP4 mesh across four four-GPU nodes.

This opt-in torchrun worker checks the physical placement before it starts any
traffic: each EP group must occupy one host, and every FSDP group must span four
hosts. It then verifies payloads for FSDP all-gather, reduce-scatter, and
all-reduce traffic, both alone and interleaved with node-local EP all-to-all.

Launch one torchrun agent per node with torchrun's ``--module`` mode. See
``README.md`` in this directory for a generic four-node Slurm command.
"""

from __future__ import annotations

import argparse
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
from tests.gpu.fault_injection.collective_payloads import run_verified_all_to_all
from tests.torchrun_process import disable_nccl_communicator_nonblocking


EXPECTED_NODES = 4
GPUS_PER_NODE = 4
WORLD_SIZE = EXPECTED_NODES * GPUS_PER_NODE
EP_SIZE = GPUS_PER_NODE
FSDP_SIZE = EXPECTED_NODES
PROCESS_GROUP_TIMEOUT_SECONDS = 180
DEFAULT_PAYLOAD_MIB = (1, 8, 32)
DEFAULT_CONTENTION_ROUNDS = 32
DEFAULT_ARRIVAL_SKEW_SECONDS = 2.0
SUCCESS_MARKER = "MULTI_NODE_EP_FSDP_TRAFFIC_OK"


@dataclass(frozen=True)
class MeshPlacement:
    rank: int
    ep_coordinate: int
    fsdp_coordinate: int
    ep_ranks: tuple[int, ...]
    fsdp_ranks: tuple[int, ...]


@dataclass(frozen=True)
class CollectiveGroup:
    process_group: dist.ProcessGroup
    ranks: tuple[int, ...]
    rank: int
    device: torch.device


@dataclass(frozen=True)
class PhaseTimings:
    fsdp_only_seconds: float
    alternating_seconds: float
    skewed_arrival_seconds: float


def _parse_payload_mib(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(","))
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("payload sizes must be positive comma-separated MiB values")
    return sizes


def _group_ranks(group: dist.ProcessGroup) -> tuple[int, ...]:
    return tuple(dist.get_process_group_ranks(group))


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

    ep_ranks = _group_ranks(mesh["ep"].get_group())
    fsdp_ranks = _group_ranks(mesh["fsdp"].get_group())
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
        ep_ranks,
        fsdp_ranks,
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


def _fsdp_all_gather(
    collective: CollectiveGroup,
    payload_mib: int,
) -> None:
    numel = _numel_for_mib(payload_mib, torch.float32)
    input_values = torch.full((numel,), float(collective.rank), device=collective.device)
    output_values = torch.empty(numel * len(collective.ranks), dtype=input_values.dtype, device=collective.device)
    dist.all_gather_into_tensor(output_values, input_values, group=collective.process_group)
    for group_index, source_rank in enumerate(collective.ranks):
        segment = output_values[group_index * numel : (group_index + 1) * numel]
        _assert_constant(segment, float(source_rank), f"FSDP all-gather source rank {source_rank}")


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
    payload_mib: tuple[int, ...],
    contention_rounds: int,
    arrival_skew_seconds: float,
) -> PhaseTimings:
    ep = CollectiveGroup(mesh["ep"].get_group(), placement.ep_ranks, placement.rank, device)
    fsdp = CollectiveGroup(mesh["fsdp"].get_group(), placement.fsdp_ranks, placement.rank, device)

    phase_start = time.monotonic()
    for size in payload_mib:
        _fsdp_all_gather(fsdp, size)
        _fsdp_reduce_scatter(fsdp, size)
        _fsdp_all_reduce(fsdp, size)
    torch.cuda.synchronize(device)
    fsdp_only_seconds = time.monotonic() - phase_start

    phase_start = time.monotonic()
    for round_index in range(contention_rounds):
        size = payload_mib[round_index % len(payload_mib)]
        run_verified_all_to_all(
            ep.process_group,
            ep.rank,
            ep.device,
            _numel_for_mib(size, torch.int64),
        )
        _fsdp_all_gather(fsdp, size)
        run_verified_all_to_all(
            ep.process_group,
            ep.rank,
            ep.device,
            _numel_for_mib(size, torch.int64),
        )
        _fsdp_reduce_scatter(fsdp, size)
    torch.cuda.synchronize(device)
    alternating_seconds = time.monotonic() - phase_start

    dist.barrier()
    phase_start = time.monotonic()
    if placement.fsdp_coordinate == placement.ep_coordinate:
        # One rank in every inter-node FSDP group arrives late. The delay is the
        # input under test, not a readiness mechanism.
        time.sleep(arrival_skew_seconds)
    _fsdp_all_gather(fsdp, max(payload_mib))
    _fsdp_reduce_scatter(fsdp, max(payload_mib))
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


def _run_with_process_group(
    payload_mib: tuple[int, ...],
    contention_rounds: int,
    arrival_skew_seconds: float,
) -> None:
    init_worker_process_group_with_device(timeout_seconds=PROCESS_GROUP_TIMEOUT_SECONDS)
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        mesh = create_device_mesh(WORLD_SIZE, fsdp_size=FSDP_SIZE, ep_size=EP_SIZE)
        placement = _validate_mesh_placement(mesh)
        dist.barrier()
        timings = _max_timings(
            _run_traffic(mesh, placement, device, payload_mib, contention_rounds, arrival_skew_seconds),
            device,
        )
        if placement.rank == 0:
            print(
                f"{SUCCESS_MARKER} world={WORLD_SIZE} nodes={EXPECTED_NODES} gpus_per_node={GPUS_PER_NODE} "
                f"ep={EP_SIZE} fsdp={FSDP_SIZE} payload_mib={','.join(map(str, payload_mib))} "
                f"rounds={contention_rounds} max_phase_seconds={timings}",
                flush=True,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(payload_mib: tuple[int, ...], contention_rounds: int, arrival_skew_seconds: float) -> None:
    disable_nccl_communicator_nonblocking(os.environ)
    # Device-mesh subgroups are created after WORLD initialization and consume
    # the production timeout environment variable, not the WORLD timeout arg.
    with patch.dict(
        os.environ,
        {"SKYRL_WORKER_NCCL_TIMEOUT_IN_S": str(PROCESS_GROUP_TIMEOUT_SECONDS)},
    ):
        _run_with_process_group(payload_mib, contention_rounds, arrival_skew_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload-mib", type=_parse_payload_mib, default=DEFAULT_PAYLOAD_MIB)
    parser.add_argument("--contention-rounds", type=int, default=DEFAULT_CONTENTION_ROUNDS)
    parser.add_argument("--arrival-skew-seconds", type=float, default=DEFAULT_ARRIVAL_SKEW_SECONDS)
    arguments = parser.parse_args()
    if arguments.contention_rounds <= 0:
        parser.error("--contention-rounds must be positive")
    if arguments.arrival_skew_seconds < 0:
        parser.error("--arrival-skew-seconds must be non-negative")
    _run(arguments.payload_mib, arguments.contention_rounds, arguments.arrival_skew_seconds)
