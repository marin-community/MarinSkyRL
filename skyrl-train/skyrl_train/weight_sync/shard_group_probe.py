"""Native custom-group qualification using the K10 expert-block schedule.

Port counters are shared device observations, not process-attributed NIC bytes.
The fixture validates expert collectives only; it does not load dense/shared weights.
"""

import hashlib
import json
import os
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
import socket
import time

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
import torch
import torch.distributed as dist

from skyrl_train.distributed.utils import get_free_port, init_custom_process_group
from skyrl_train.weight_sync.readback_diagnostics import ENVIRONMENT_KEYS, network_log_readback
from skyrl_train.weight_sync.shard_group_schedule import (
    ExpertEntry,
    ReceiverRank,
    ShardGroupSchedule,
    TrainerRank,
    build_shard_group_schedule,
)


def tiny_schedule(receiver_replicas: int, payload_bytes: int, *, ep: int = 1) -> ShardGroupSchedule:
    """Two PP owners per expert block, with every inference replica a member."""
    if receiver_replicas not in (2, 4):
        raise ValueError("The native fixture requires two or four receiver replicas")
    trainers = tuple(TrainerRank(pp * ep + k, 0, pp, k) for pp in range(2) for k in range(ep))
    receivers = tuple(
        ReceiverRank(replica * ep + k, replica, k) for replica in range(receiver_replicas) for k in range(ep)
    )
    entries = tuple(
        ExpertEntry(f"layer{layer}.expert{expert}.{projection}", layer, layer, expert, projection, payload_bytes)
        for layer in range(2)
        for expert in range(ep)
        for projection in ("fc1", "fc2")
    )
    return build_shard_group_schedule(
        trainers, receivers, entries, trainer_ep=ep, receiver_ep=ep, layers_by_pp=((0,), (1,)), num_experts=ep
    )


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


class ShardProbeRank:
    def __init__(self, rank: int, schedule: ShardGroupSchedule, backend: str, output: str):
        self.rank = rank
        self.schedule = schedule
        self.backend = backend
        self.device = torch.device("cuda", 0) if backend == "nccl" else torch.device("cpu")
        self.groups = {}
        self.rows = []
        self.output = Path(output) / f"rank-{rank}.json"
        self.output.parent.mkdir(parents=True, exist_ok=True)
        if backend == "nccl":
            torch.cuda.set_device(self.device)
        self.record("before_default_group")
        dist.init_process_group(
            backend,
            init_method=f"tcp://127.0.0.1:{get_free_port()}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=30),
            device_id=self.device if backend == "nccl" else None,
        )

    def record(self, stage: str, **values) -> dict:
        row = {
            "stage": stage,
            "rank": self.rank,
            "pid": os.getpid(),
            "physical_node": os.environ.get("IRIS_NODE_NAME"),
            "host": socket.gethostname(),
            "ray_node_id": ray.get_runtime_context().get_node_id(),
            "monotonic": time.monotonic(),
            **values,
        }
        self.rows.append(row)
        self.output.write_text(json.dumps(self.rows, sort_keys=True))
        return row

    def describe(self) -> dict:
        device = None
        if self.backend == "nccl":
            properties = torch.cuda.get_device_properties(self.device)
            device = {"name": properties.name, "uuid": str(properties.uuid), "local_ordinal": self.device.index}
        return self.record(
            "ready",
            default_world=dist.get_world_size(),
            default_rank=dist.get_rank(),
            environment={key: os.environ.get(key) for key in ENVIRONMENT_KEYS},
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            device=device,
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            address=ray.util.get_node_ip_address(),
            port=get_free_port(),
            counters=port_counters(),
        )

    def initialize(self, ep: int, address: str, port: int) -> dict:
        group = self.schedule.groups[ep]
        local_rank = group.members.index(self.rank)
        self.groups[ep] = init_custom_process_group(
            backend=self.backend,
            init_method=f"tcp://{address}:{port}",
            world_size=len(group.members),
            rank=local_rank,
            group_name=f"shard-probe-ep{ep}",
            timeout=timedelta(seconds=30),
        )
        return self.record("group_ready", ep=ep, members=group.members, group_rank=local_rank)

    def transfer(self, index: int) -> dict:
        item = self.schedule.broadcasts[index]
        members = self.schedule.groups[item.group_ep].members
        group_root = members.index(item.root)
        tensor = torch.empty(item.nbytes, dtype=torch.uint8, device=self.device)
        value = (index * 17 + 3) % 251
        tensor.fill_(value if self.rank == item.root else 255)
        if self.backend == "nccl":
            torch.cuda.synchronize()
        before = port_counters()
        started = time.monotonic()
        # init_custom_process_group registers local identity ranks. Passing the
        # schedule's global root directly is incorrect for EP blocks after zero.
        dist.broadcast(tensor, src=group_root, group=self.groups[item.group_ep])
        if self.backend == "nccl":
            torch.cuda.synchronize()
        elapsed = time.monotonic() - started
        after = port_counters()
        raw = tensor.cpu().numpy().tobytes()
        if raw != bytes([value]) * item.nbytes:
            raise ValueError("Expert-block payload mismatch")
        return self.record(
            "broadcast",
            index=index,
            entry=asdict(item),
            group_root=group_root,
            received_bytes=len(raw),
            sha256=hashlib.sha256(raw).hexdigest(),
            seconds=elapsed,
            counters_before=before,
            counters_after=after,
        )

    def finish(self) -> dict:
        for group in self.groups.values():
            dist.destroy_process_group(group)
        self.groups.clear()
        if dist.is_initialized():
            dist.destroy_process_group()
        return self.record("closed", network_log=network_log_readback())


def validate_role_hosts(rows: list[dict], trainer_count: int) -> None:
    """Require one physical/Ray host per role and distinct hosts across roles."""
    identities = []
    for role in (rows[:trainer_count], rows[trainer_count:]):
        if not role or any(not row.get("physical_node") or not row.get("ray_node_id") for row in role):
            raise ValueError("Missing physical host or Ray node identity")
        physical = {row["physical_node"] for row in role}
        nodes = {row["ray_node_id"] for row in role}
        if len(physical) != 1 or len(nodes) != 1:
            raise ValueError("Each role requires one physical host and Ray node")
        identities.append((physical.pop(), nodes.pop()))
    if identities[0][0] == identities[1][0] or identities[0][1] == identities[1][1]:
        raise ValueError("Trainer and receiver roles require distinct physical hosts and Ray nodes")


def run_group_probe(
    schedule: ShardGroupSchedule, backend: str, output: str, measurement_started=None, *, two_hosts: bool = False
) -> dict:
    """Execute every scheduled collective through real Ray actors and custom PGs."""
    if backend not in ("gloo", "nccl") or max(item.nbytes for item in schedule.broadcasts) > 32 * 1024**2:
        raise ValueError("Unsupported native fixture backend or allocation")
    if backend == "nccl" and not two_hosts:
        raise ValueError("The native NCCL fixture requires explicit two-host placement")
    result = {"schedule": asdict(schedule), "ready": [], "groups": [], "broadcasts": [], "cleanup": [], "error": None}
    actors = []
    try:
        actor_type = ray.remote(num_cpus=1, num_gpus=1 if backend == "nccl" else 0)(ShardProbeRank)
        trainer_count = len(schedule.trainer_global_ranks)
        count = trainer_count + len(schedule.receiver_global_ranks)
        nodes = []
        if two_hosts:
            resource, minimum = ("GPU", 8) if backend == "nccl" else ("CPU", max(trainer_count, count - trainer_count))
            nodes = sorted(
                row["NodeID"] for row in ray.nodes() if row["Alive"] and row["Resources"].get(resource, 0) >= minimum
            )
            if len(nodes) != 2:
                raise ValueError("Expected exactly two qualified native Ray nodes")
        for rank in range(count):
            selected = actor_type
            if two_hosts:
                selected = selected.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(nodes[int(rank >= trainer_count)], soft=False)
                )
            actors.append(selected.remote(rank, schedule, backend, output))
        ready = ray.get([actor.describe.remote() for actor in actors], timeout=90)
        result["ready"] = ready
        if two_hosts:
            validate_role_hosts(ready, trainer_count)
        if any(row["default_world"] != 1 or row["default_rank"] != 0 for row in ready):
            raise ValueError("Expected independent default worlds")
        if any(row["environment"] != ready[0]["environment"] for row in ready):
            raise ValueError("Communicator environments differ")
        if ready[0]["environment"]["VLLM_BATCH_INVARIANT"] not in (None, "0"):
            raise ValueError("Native timing requires batch invariance off")
        for group in schedule.groups:
            root = ready[group.members[0]]
            result["groups"].extend(
                ray.get(
                    [actors[rank].initialize.remote(group.ep, root["address"], root["port"]) for rank in group.members],
                    timeout=60,
                )
            )
        if measurement_started is not None:
            measurement_started(result)
        for index, item in enumerate(schedule.broadcasts):
            result["broadcasts"].extend(
                ray.get(
                    [actors[rank].transfer.remote(index) for rank in schedule.groups[item.group_ep].members], timeout=60
                )
            )
    except Exception as error:
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        try:
            result["cleanup"] = ray.get([actor.finish.remote() for actor in actors], timeout=45)
        except Exception as error:
            result["cleanup_error"] = f"{type(error).__name__}: {error}"
        for actor in actors:
            ray.kill(actor, no_restart=True)
        Path(output).mkdir(parents=True, exist_ok=True)
        Path(output, "result.json").write_text(json.dumps(result, sort_keys=True))
    return result
