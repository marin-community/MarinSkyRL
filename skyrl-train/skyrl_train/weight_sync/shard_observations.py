"""Physical port and GPU observations; shared counters are not process traffic."""

import csv
import io
import os
from pathlib import Path
import subprocess
import socket
import time

import ray
import torch

from skyrl_train.weight_sync.readback_diagnostics import ENVIRONMENT_KEYS, persist_readback
from skyrl_train.weight_sync.shard_memory import device_memory


def port_counters(root: Path = Path("/sys/class/infiniband")) -> dict:
    """Read the standard IB data counters, whose units are four octets.

    Each process retains all visible ports. Shared-port observations must be
    deduplicated by physical host/device/port; they cannot establish per-sender
    egress without an independently verified exclusive process-to-HCA binding.
    """
    rows = []
    for directory in sorted(root.glob("*/ports/*/counters")):
        row = {"device": directory.parents[2].name, "port": directory.parent.name}
        try:
            row.update(
                {name: int((directory / name).read_text().strip()) for name in ("port_xmit_data", "port_rcv_data")}
            )
            row["counter_unit_bytes"] = 4
        except (OSError, ValueError) as error:
            row["error"] = f"{type(error).__name__}: {error}"
        rows.append(row)
    return {"ports": rows, "attribution": "shared-port, not process", "observed_monotonic": time.monotonic()}


def hardware_identity(device, root=Path("/sys/class/infiniband")):
    hcas = []
    for hca in sorted(root.iterdir()) if root.exists() else ():
        hcas.append({"device": hca.name, "pci_bus_id": (hca / "device").resolve().name})
    result = {
        "physical_node": os.environ.get("IRIS_NODE_NAME"),
        "ray_node_id": ray.get_runtime_context().get_node_id() if ray.is_initialized() else None,
        "hcas": hcas,
        "environment": {
            key: os.environ.get(key)
            for key in (*ENVIRONMENT_KEYS, "CUDA_DEVICE_MAX_CONNECTIONS", "CUDA_VISIBLE_DEVICES")
        },
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "cuda_measured": device.type == "cuda",
        "binding_scope": "observed GPU/HCA PCI identities; no exclusive process-to-port binding inferred",
    }
    if device.type == "cuda":
        result["gpu_uuid"] = str(torch.cuda.get_device_properties(device).uuid)
        try:
            output = subprocess.run(
                ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout
            rows = {
                uuid.strip().removeprefix("GPU-"): (uuid.strip(), bus.strip())
                for uuid, bus in csv.reader(io.StringIO(output))
            }
            raw_uuid, bus = rows[result["gpu_uuid"].removeprefix("GPU-")]
            result["gpu_nvidia_smi_uuid"] = raw_uuid
            result["gpu_pci_bus_id"] = bus
        except (OSError, subprocess.SubprocessError, KeyError, ValueError) as error:
            result["gpu_pci_error"] = f"{type(error).__name__}: {error}"
    return result


def observe_worker(worker, device, role, observation_id, output_uri):
    """Persist one physical endpoint outside the matched reference's broadcast timer."""
    identity = getattr(worker, "_physical_weight_sync_identity", None)
    if identity is None:
        identity = {
            **hardware_identity(device),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "role": role,
            "rank": torch.distributed.get_rank(),
            "world_size": torch.distributed.get_world_size(),
            "attempt_uid": os.environ.get("IRIS_ATTEMPT_UID"),
            "task_id": os.environ.get("IRIS_TASK_ID"),
        }
        worker._physical_weight_sync_identity = identity
    memory = device_memory(device)
    result = {"observation_id": observation_id, "identity": identity, "ports": port_counters(), "memory": memory}
    return {**result, "durable_receipt": persist_readback(output_uri, f"physical-{observation_id}-{role}", result)}
