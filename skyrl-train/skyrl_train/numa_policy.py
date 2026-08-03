"""Dependency-light Linux NUMA memory-policy support.

Ray imports this module before assigning actor-specific CUDA visibility. Keep it
free of Ray, PyTorch, and package imports that transitively load either runtime.
"""

import os
import re
from ctypes import CDLL, POINTER, Structure, byref, c_char_p, c_int, c_ulong, c_void_p, get_errno, sizeof
from ctypes.util import find_library
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


_MEMORY_POLICY_NAMES = {
    0: "default",
    1: "preferred",
    2: "bind",
    3: "interleave",
    4: "local",
    5: "preferred-many",
}


@dataclass(frozen=True)
class MemoryPolicy:
    """Effective NUMA task policy for the calling thread."""

    mode: str
    nodes: tuple[int, ...]


class _Bitmask(Structure):
    _fields_ = [("size", c_ulong), ("maskp", POINTER(c_ulong))]


def is_numa_affinity_enabled() -> bool:
    """Return whether explicit NUMA placement is enabled."""
    return os.environ.get("SKYRL_ENABLE_NUMA_AFFINITY", "0") == "1"


def parse_numa_range_list(value: str) -> list[int]:
    """Expand a Linux NUMA range list such as ``0-3,12``."""
    if not value:
        return []
    values = []
    for part in value.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def load_libnuma() -> CDLL:
    """Load libnuma with syscall errno capture enabled."""
    library_path = find_library("numa")
    if library_path is None:
        raise RuntimeError("NUMA memory policy requires libnuma")
    return CDLL(library_path, use_errno=True)


def current_memory_policy() -> MemoryPolicy:
    """Return the calling thread's effective Linux NUMA task policy."""
    libnuma = load_libnuma()
    libnuma.numa_max_node.argtypes = []
    libnuma.numa_max_node.restype = c_int
    max_node = libnuma.numa_max_node()
    if max_node < 0:
        raise RuntimeError("libnuma could not determine the maximum NUMA node")

    bits_per_word = 8 * sizeof(c_ulong)
    word_count = (max_node + 1 + bits_per_word - 1) // bits_per_word
    mask = (c_ulong * word_count)()
    mode = c_int()
    libnuma.get_mempolicy.argtypes = [POINTER(c_int), POINTER(c_ulong), c_ulong, c_void_p, c_ulong]
    libnuma.get_mempolicy.restype = c_int
    if libnuma.get_mempolicy(byref(mode), mask, max_node + 1, None, 0) != 0:
        errno = get_errno()
        raise OSError(errno, os.strerror(errno))

    nodes = tuple(node for node in range(max_node + 1) if mask[node // bits_per_word] & (1 << (node % bits_per_word)))
    return MemoryPolicy(mode=_MEMORY_POLICY_NAMES.get(mode.value, f"unknown-{mode.value}"), nodes=nodes)


def cpu_numa_topology(node_root: Path = Path("/sys/devices/system/node")) -> dict[int, tuple[int, ...]]:
    """Return each CPU-bearing NUMA node and the logical CPUs it contains."""
    topology = {}
    try:
        node_paths = node_root.iterdir()
    except OSError as error:
        raise RuntimeError("could not discover CPU-bearing NUMA nodes") from error
    for node_path in node_paths:
        match = re.fullmatch(r"node(\d+)", node_path.name)
        if match is None:
            continue
        try:
            cpus = tuple(parse_numa_range_list((node_path / "cpulist").read_text().strip()))
        except (OSError, ValueError) as error:
            raise RuntimeError(f"could not read NUMA CPU list from {node_path / 'cpulist'}") from error
        if cpus:
            topology[int(match.group(1))] = cpus
    if not topology:
        raise RuntimeError("could not discover CPU-bearing NUMA nodes")
    return dict(sorted(topology.items()))


def allowed_memory_nodes(status_path: Path = Path("/proc/self/status")) -> set[int]:
    """Return the NUMA nodes allowed by the process's cpuset cgroup."""
    with status_path.open() as status_file:
        for line in status_file:
            if line.startswith("Mems_allowed_list:"):
                return set(parse_numa_range_list(line.split(":", 1)[1].strip()))
    raise RuntimeError(f"{status_path} does not expose Mems_allowed_list")


def host_memory_nodes(cpu_topology: dict[int, tuple[int, ...]], allowed_nodes: set[int]) -> tuple[int, ...]:
    """Select allowed CPU-bearing nodes, excluding memory-only GPU nodes."""
    nodes = tuple(sorted(set(cpu_topology).intersection(allowed_nodes)))
    if not nodes:
        raise RuntimeError(
            f"no CPU-bearing NUMA nodes are allowed: cpu_nodes={sorted(cpu_topology)} "
            f"allowed_nodes={sorted(allowed_nodes)}"
        )
    return nodes


def set_host_memory_policy() -> Optional[tuple[int, ...]]:
    """Restrict this thread and its future children to host-memory NUMA nodes."""
    if not is_numa_affinity_enabled():
        return None

    target_memory_nodes = host_memory_nodes(cpu_numa_topology(), allowed_memory_nodes())
    _set_membind(target_memory_nodes)
    return target_memory_nodes


def _set_membind(nodes: tuple[int, ...]) -> None:
    """Restrict future allocations to CPU-bearing NUMA nodes."""
    libnuma = load_libnuma()
    libnuma.numa_parse_nodestring.argtypes = [c_char_p]
    libnuma.numa_parse_nodestring.restype = POINTER(_Bitmask)
    libnuma.numa_set_membind.argtypes = [POINTER(_Bitmask)]
    libnuma.numa_set_membind.restype = None
    libnuma.numa_bitmask_free.argtypes = [POINTER(_Bitmask)]
    libnuma.numa_bitmask_free.restype = None

    mask = libnuma.numa_parse_nodestring(",".join(str(node) for node in nodes).encode())
    if not mask:
        raise RuntimeError(f"libnuma could not parse host-memory nodes {nodes}")
    try:
        libnuma.numa_set_membind(mask)
    finally:
        libnuma.numa_bitmask_free(mask)

    policy = current_memory_policy()
    if policy.mode != "bind" or policy.nodes != nodes:
        raise RuntimeError(f"failed to install host-memory policy: requested={nodes} effective={policy}")
