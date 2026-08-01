"""Exercise the production EP4/FSDP4 mesh across four four-GPU nodes.

This opt-in torchrun worker checks the physical placement before it starts any
traffic: each EP group must occupy one host, and every FSDP group must span four
hosts. It then verifies payloads for FSDP all-gather, reduce-scatter, and
all-reduce traffic, both alone and interleaved with node-local EP all-to-all.

Launch one torchrun agent per node. See ``README.md`` in this directory for a
generic four-node Slurm command.
"""

from __future__ import annotations

import argparse
import os
import socket
import time
from collections import Counter
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh

from skyrl_train.distributed.fsdp_utils import create_device_mesh
from skyrl_train.distributed.utils import init_worker_process_group_with_device


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
NCCL_COMMUNICATOR_NONBLOCKING_ENVIRONMENT = (
    "TORCH_NCCL_USE_COMM_NONBLOCKING",
    "TORCH_NCCL_NONBLOCKING_TIMEOUT",
)


@dataclass(frozen=True)
class MeshPlacement:
    rank: int
    local_rank: int
    hostname: str
    ep_coordinate: int
    fsdp_coordinate: int
    ep_ranks: tuple[int, ...]
    fsdp_ranks: tuple[int, ...]


def _parse_payload_mib(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item) for item in value.split(","))
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("payload sizes must be positive comma-separated MiB values")
    return sizes


def _group_ranks(group: dist.ProcessGroup) -> tuple[int, ...]:
    return tuple(dist.get_process_group_ranks(group))


def _validate_mesh_placement(mesh: DeviceMesh) -> MeshPlacement:
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
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
        local_rank,
        hostname,
        coordinates["ep"],
        coordinates["fsdp"],
        ep_ranks,
        fsdp_ranks,
    )


def _numel_for_mib(payload_mib: int, dtype: torch.dtype) -> int:
    element_size = torch.empty((), dtype=dtype).element_size()
    return payload_mib * 1024 * 1024 // element_size


def _assert_constant(tensor: torch.Tensor, expected: float, context: str) -> None:
    minimum, maximum = torch.aminmax(tensor)
    if minimum.item() != expected or maximum.item() != expected:
        raise AssertionError(
            f"{context}: expected every value to be {expected}, got min={minimum.item()} max={maximum.item()}"
        )


def _fsdp_all_gather(
    group: dist.ProcessGroup,
    group_ranks: tuple[int, ...],
    rank: int,
    device: torch.device,
    payload_mib: int,
) -> None:
    numel = _numel_for_mib(payload_mib, torch.float32)
    input_values = torch.full((numel,), float(rank), device=device)
    output_values = torch.empty(numel * len(group_ranks), dtype=input_values.dtype, device=device)
    dist.all_gather_into_tensor(output_values, input_values, group=group)
    for group_index, source_rank in enumerate(group_ranks):
        segment = output_values[group_index * numel : (group_index + 1) * numel]
        _assert_constant(segment, float(source_rank), f"FSDP all-gather source rank {source_rank}")


def _fsdp_reduce_scatter(
    group: dist.ProcessGroup,
    group_ranks: tuple[int, ...],
    rank: int,
    device: torch.device,
    payload_mib: int,
) -> None:
    numel = _numel_for_mib(payload_mib, torch.float32)
    chunks = [
        torch.full((numel,), float(rank + destination_index), device=device)
        for destination_index in range(len(group_ranks))
    ]
    input_values = torch.cat(chunks)
    output_values = torch.empty(numel, dtype=input_values.dtype, device=device)
    dist.reduce_scatter_tensor(output_values, input_values, group=group)
    destination_index = group_ranks.index(rank)
    expected = float(sum(group_ranks) + destination_index * len(group_ranks))
    _assert_constant(output_values, expected, f"FSDP reduce-scatter destination rank {rank}")


def _fsdp_all_reduce(
    group: dist.ProcessGroup,
    group_ranks: tuple[int, ...],
    rank: int,
    device: torch.device,
    payload_mib: int,
) -> None:
    numel = _numel_for_mib(payload_mib, torch.float32)
    values = torch.full((numel,), float(rank), device=device)
    dist.all_reduce(values, group=group)
    _assert_constant(values, float(sum(group_ranks)), f"FSDP all-reduce rank {rank}")


def _ep_all_to_all(
    group: dist.ProcessGroup,
    group_ranks: tuple[int, ...],
    rank: int,
    device: torch.device,
    payload_mib: int,
) -> None:
    numel = _numel_for_mib(payload_mib, torch.float32)
    values_per_peer = max(1, numel // len(group_ranks))
    group_rank = group_ranks.index(rank)
    chunks = [
        torch.full((values_per_peer,), float(rank * 100 + destination_index), device=device)
        for destination_index in range(len(group_ranks))
    ]
    input_values = torch.cat(chunks)
    output_values = torch.empty_like(input_values)
    dist.all_to_all_single(output_values, input_values, group=group)
    for source_index, source_rank in enumerate(group_ranks):
        segment = output_values[source_index * values_per_peer : (source_index + 1) * values_per_peer]
        _assert_constant(segment, float(source_rank * 100 + group_rank), f"EP all-to-all source rank {source_rank}")


def _run_traffic(
    mesh: DeviceMesh,
    placement: MeshPlacement,
    device: torch.device,
    payload_mib: tuple[int, ...],
    contention_rounds: int,
    arrival_skew_seconds: float,
) -> dict[str, float]:
    ep_group = mesh["ep"].get_group()
    fsdp_group = mesh["fsdp"].get_group()
    timings: dict[str, float] = {}

    phase_start = time.monotonic()
    for size in payload_mib:
        _fsdp_all_gather(fsdp_group, placement.fsdp_ranks, placement.rank, device, size)
        _fsdp_reduce_scatter(fsdp_group, placement.fsdp_ranks, placement.rank, device, size)
        _fsdp_all_reduce(fsdp_group, placement.fsdp_ranks, placement.rank, device, size)
    torch.cuda.synchronize(device)
    timings["fsdp_only"] = time.monotonic() - phase_start

    phase_start = time.monotonic()
    for round_index in range(contention_rounds):
        size = payload_mib[round_index % len(payload_mib)]
        _ep_all_to_all(ep_group, placement.ep_ranks, placement.rank, device, size)
        _fsdp_all_gather(fsdp_group, placement.fsdp_ranks, placement.rank, device, size)
        _ep_all_to_all(ep_group, placement.ep_ranks, placement.rank, device, size)
        _fsdp_reduce_scatter(fsdp_group, placement.fsdp_ranks, placement.rank, device, size)
    torch.cuda.synchronize(device)
    timings["alternating"] = time.monotonic() - phase_start

    dist.barrier()
    phase_start = time.monotonic()
    if placement.fsdp_coordinate == placement.ep_coordinate:
        # One rank in every inter-node FSDP group arrives late. The delay is the
        # input under test, not a readiness mechanism.
        time.sleep(arrival_skew_seconds)
    _fsdp_all_gather(fsdp_group, placement.fsdp_ranks, placement.rank, device, max(payload_mib))
    _fsdp_reduce_scatter(fsdp_group, placement.fsdp_ranks, placement.rank, device, max(payload_mib))
    torch.cuda.synchronize(device)
    timings["skewed_arrival"] = time.monotonic() - phase_start
    return timings


def _max_timings(local_timings: dict[str, float], device: torch.device) -> dict[str, float]:
    names = tuple(local_timings)
    values = torch.tensor([local_timings[name] for name in names], dtype=torch.float64, device=device)
    dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return dict(zip(names, values.cpu().tolist(), strict=True))


def _run(payload_mib: tuple[int, ...], contention_rounds: int, arrival_skew_seconds: float) -> None:
    for variable in NCCL_COMMUNICATOR_NONBLOCKING_ENVIRONMENT:
        os.environ.pop(variable, None)
    os.environ["SKYRL_WORKER_NCCL_TIMEOUT_IN_S"] = str(PROCESS_GROUP_TIMEOUT_SECONDS)
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
