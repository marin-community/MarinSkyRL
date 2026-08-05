"""Shared construction and validation for the four-node EP4/FSDP4 mesh."""

from __future__ import annotations

import socket
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DeviceMesh

from skyrl_train.distributed.fsdp_utils import create_device_mesh
from skyrl_train.distributed.utils import init_worker_process_group_with_device
from tests.gpu.fault_injection.collective_payloads import MeshCollectives
from tests.gpu.fault_injection.multi_node_geometry import (
    EP_SIZE,
    EXPECTED_NODES,
    FSDP_SIZE,
    GPUS_PER_NODE,
    WORLD_SIZE,
)


@dataclass(frozen=True)
class MeshPlacement:
    rank: int
    ep_coordinate: int
    fsdp_coordinate: int


@dataclass(frozen=True)
class MeshRuntime:
    mesh: DeviceMesh
    placement: MeshPlacement
    device: torch.device

    def collective_groups(self) -> MeshCollectives:
        return MeshCollectives.from_mesh(self.mesh, self.placement.rank, self.device)


def _validate_world_geometry(process_environment: Mapping[str, str]) -> None:
    local_world_size = int(process_environment["LOCAL_WORLD_SIZE"])
    if dist.get_world_size() != WORLD_SIZE:
        raise AssertionError(f"expected {WORLD_SIZE} ranks, got {dist.get_world_size()}")
    if local_world_size != GPUS_PER_NODE:
        raise AssertionError(f"expected {GPUS_PER_NODE} ranks per node, got {local_world_size}")


def _rank_hostnames(hostname: str) -> tuple[str, ...]:
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
    return resolved_hostnames


def _mesh_placement(mesh: DeviceMesh, resolved_hostnames: tuple[str, ...], hostname: str) -> MeshPlacement:
    rank = dist.get_rank()
    ep_ranks = tuple(dist.get_process_group_ranks(mesh["ep"].get_group()))
    fsdp_ranks = tuple(dist.get_process_group_ranks(mesh["fsdp"].get_group()))
    coordinates = dict(zip(mesh.mesh_dim_names, mesh.get_coordinate(), strict=True))
    ep_hosts = {resolved_hostnames[group_rank] for group_rank in ep_ranks}
    fsdp_hosts = {resolved_hostnames[group_rank] for group_rank in fsdp_ranks}
    if ep_hosts != {hostname}:
        raise AssertionError(f"EP group {ep_ranks} crosses hosts: {sorted(ep_hosts)}")
    if len(fsdp_hosts) != EXPECTED_NODES:
        raise AssertionError(f"FSDP group {fsdp_ranks} spans {len(fsdp_hosts)} hosts: {sorted(fsdp_hosts)}")

    return MeshPlacement(rank, coordinates["ep"], coordinates["fsdp"])


@contextmanager
def multi_node_mesh_runtime(
    timeout_seconds: int,
    process_environment: Mapping[str, str],
) -> Iterator[MeshRuntime]:
    """Yield the validated production mesh and always destroy its world group."""

    init_worker_process_group_with_device(timeout_seconds=timeout_seconds)
    try:
        device = torch.device("cuda", int(process_environment["LOCAL_RANK"]))
        _validate_world_geometry(process_environment)
        mesh = create_device_mesh(
            WORLD_SIZE,
            fsdp_size=FSDP_SIZE,
            ep_size=EP_SIZE,
            timeout_seconds=timeout_seconds,
        )
        hostname = socket.gethostname()
        runtime = MeshRuntime(mesh, _mesh_placement(mesh, _rank_hostnames(hostname), hostname), device)
        dist.barrier()
        yield runtime
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
