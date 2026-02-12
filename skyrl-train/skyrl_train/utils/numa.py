"""GPU NUMA affinity utilities for multi-socket and unified memory architectures (e.g., GH200).

On GH200 nodes, each GPU has its own NUMA node for HBM memory (e.g., nodes 4, 12, 20, 28)
which is separate from the CPU NUMA node (0, 1, 2, 3). This module detects the correct
CPU NUMA affinity for each GPU and binds the calling process accordingly.

Activation: Set SKYRL_ENABLE_NUMA_AFFINITY=1 in the environment. When unset, all functions
are no-ops to avoid interfering with systems that don't need NUMA binding.
"""

import os
import re
import subprocess
from ctypes import CDLL, Structure, POINTER, c_ulong, c_char_p, c_int, c_void_p
from ctypes.util import find_library
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from loguru import logger


def is_numa_affinity_enabled() -> bool:
    """Check if NUMA affinity binding is enabled via environment variable."""
    return os.environ.get("SKYRL_ENABLE_NUMA_AFFINITY", "0") == "1"


@lru_cache(maxsize=1)
def _parse_nvidia_smi_topo() -> Optional[Dict[int, Tuple[List[int], int]]]:
    """Parse nvidia-smi topo -m to get GPU -> (cpu_list, numa_node) mapping.

    Returns:
        Dict mapping GPU index to (cpu_affinity_list, cpu_numa_node), or None on failure.
        Example on GH200: {0: ([0..71], 0), 1: ([72..143], 1), ...}
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    gpu_map = {}
    for line in result.stdout.splitlines():
        # Match lines starting with "GPU0", "GPU1", etc.
        match = re.match(r"^GPU(\d+)\s+", line)
        if not match:
            continue

        gpu_idx = int(match.group(1))

        # Split by whitespace — the CPU Affinity and NUMA Affinity columns
        # are the last two (or three with GPU NUMA ID) tab-separated fields.
        # Format: GPU0 <connections...> <CPU Affinity> <NUMA Affinity> [<GPU NUMA ID>]
        parts = line.split()

        # Find CPU Affinity field: looks like "0-71" or "0-71,144-215"
        cpu_affinity_str = None
        numa_node = None
        for i, part in enumerate(parts):
            if re.match(r"^\d+(-\d+)?(,\d+(-\d+)?)*$", part) and "-" in part:
                cpu_affinity_str = part
                # Next numeric field after CPU Affinity is NUMA Affinity
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    numa_node = int(parts[i + 1])
                break

        if cpu_affinity_str is not None and numa_node is not None:
            cpus = _parse_cpu_range(cpu_affinity_str)
            gpu_map[gpu_idx] = (cpus, numa_node)

    return gpu_map if gpu_map else None


def _parse_cpu_range(cpu_str: str) -> List[int]:
    """Parse CPU range string like '0-71' or '0-71,144-215' into list of CPU IDs."""
    cpus = []
    for part in cpu_str.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            cpus.extend(range(int(start), int(end) + 1))
        else:
            cpus.append(int(part))
    return cpus


@lru_cache(maxsize=1)
def _get_sysfs_gpu_numa_map() -> Optional[Dict[int, int]]:
    """Fallback: Get GPU -> CPU NUMA node mapping from sysfs.

    Reads /sys/bus/pci/devices/<bus_id>/numa_node for each GPU.
    On GH200, this returns the GPU's HBM NUMA node (4, 12, 20, 28),
    NOT the CPU NUMA node. We then need to find the closest CPU NUMA node.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    gpu_numa_map = {}
    for line in result.stdout.strip().splitlines():
        parts = line.split(",")
        if len(parts) != 2:
            continue
        gpu_idx = int(parts[0].strip())
        bus_id = parts[1].strip().lower()

        # sysfs uses lowercase without leading domain zeros sometimes
        # Try multiple path formats
        numa_node = _read_pci_numa_node(bus_id)
        if numa_node is not None:
            gpu_numa_map[gpu_idx] = numa_node

    return gpu_numa_map if gpu_numa_map else None


def _read_pci_numa_node(bus_id: str) -> Optional[int]:
    """Read NUMA node from sysfs for a PCI device."""
    # nvidia-smi reports bus_id like "00000009:01:00.0"
    # sysfs paths may use different formats
    candidates = [bus_id, bus_id.lstrip("0") if bus_id.startswith("0") else bus_id]
    for bid in candidates:
        path = f"/sys/bus/pci/devices/{bid}/numa_node"
        try:
            with open(path) as f:
                val = int(f.read().strip())
                if val >= 0:
                    return val
        except (FileNotFoundError, ValueError):
            continue
    return None


def get_gpu_cpu_affinity(gpu_physical_id: int) -> Optional[Tuple[List[int], int]]:
    """Get the optimal CPU list and NUMA node for a GPU.

    Args:
        gpu_physical_id: Physical GPU index (as seen by nvidia-smi, not CUDA logical index).

    Returns:
        Tuple of (cpu_list, numa_node) or None if detection fails.
    """
    # Primary: nvidia-smi topo (most reliable, handles GH200 correctly)
    topo = _parse_nvidia_smi_topo()
    if topo and gpu_physical_id in topo:
        return topo[gpu_physical_id]

    # Fallback: sysfs (may return GPU NUMA node instead of CPU NUMA node on GH200)
    sysfs_map = _get_sysfs_gpu_numa_map()
    if sysfs_map and gpu_physical_id in sysfs_map:
        numa_node = sysfs_map[gpu_physical_id]
        # Try to get CPU list for this NUMA node
        cpu_list_path = f"/sys/devices/system/node/node{numa_node}/cpulist"
        try:
            with open(cpu_list_path) as f:
                cpus = _parse_cpu_range(f.read().strip())
                if cpus:
                    return (cpus, numa_node)
        except (FileNotFoundError, ValueError):
            pass

    return None


def set_numa_affinity_for_gpu(gpu_id: int) -> None:
    """Set CPU and memory affinity for the calling process to match a GPU's NUMA node.

    This binds the process to the CPUs closest to the specified GPU and sets the
    memory allocation policy to prefer the GPU's NUMA node.

    Args:
        gpu_id: Physical GPU index.

    No-op if:
        - SKYRL_ENABLE_NUMA_AFFINITY != "1"
        - GPU NUMA topology cannot be detected
        - libnuma is not available
    """
    if not is_numa_affinity_enabled():
        return

    affinity = get_gpu_cpu_affinity(gpu_id)
    if affinity is None:
        logger.debug(f"NUMA affinity: could not detect topology for GPU {gpu_id}")
        return

    cpu_list, numa_node = affinity

    # Try os.sched_setaffinity first (doesn't need libnuma)
    try:
        os.sched_setaffinity(0, cpu_list)
        logger.info(
            f"NUMA affinity: bound process to CPUs {cpu_list[0]}-{cpu_list[-1]} "
            f"(NUMA node {numa_node}) for GPU {gpu_id}"
        )
    except (OSError, AttributeError) as e:
        logger.debug(f"NUMA affinity: os.sched_setaffinity failed: {e}")
        return

    # Also set memory binding via libnuma if available
    _set_membind_via_libnuma(numa_node)


def _set_membind_via_libnuma(numa_node: int) -> None:
    """Set memory binding policy to prefer the specified NUMA node using libnuma."""
    try:
        libnuma = CDLL(find_library("numa"))
    except (OSError, TypeError):
        # libnuma not installed — CPU affinity is already set, memory binding is optional
        return

    class bitmask_t(Structure):
        _fields_ = [
            ("size", c_ulong),
            ("maskp", POINTER(c_ulong)),
        ]

    try:
        libnuma.numa_parse_nodestring.argtypes = [c_char_p]
        libnuma.numa_parse_nodestring.restype = POINTER(bitmask_t)
        libnuma.numa_set_preferred.argtypes = [c_int]
        libnuma.numa_set_preferred.restype = None

        # Use numa_set_preferred (soft) instead of numa_set_membind (hard)
        # to avoid OOM when the local NUMA node is full
        libnuma.numa_set_preferred(c_int(numa_node))
        logger.debug(f"NUMA affinity: set memory preferred to NUMA node {numa_node}")
    except Exception as e:
        logger.debug(f"NUMA affinity: libnuma membind failed: {e}")
