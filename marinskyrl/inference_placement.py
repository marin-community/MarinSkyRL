"""Node-local inference placement contracts shared by launch and runtime."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


def validate_node_local_inference(
    *,
    enabled: bool,
    backend: str,
    async_engine: bool,
    colocate_all: bool,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    num_inference_engines: int,
    gpus_per_node: int | None = None,
    remote: bool = False,
) -> None:
    """Reject unsupported opt-ins without changing legacy placement contracts."""
    if not enabled:
        return
    if backend != "vllm" or not async_engine or colocate_all or remote:
        raise ValueError("inference_engine_node_local requires local, non-colocated async vLLM engines")
    if tensor_parallel_size != 1 or pipeline_parallel_size != 1:
        raise ValueError("inference_engine_node_local currently requires TP=PP=1")
    if data_parallel_size <= 0 or num_inference_engines <= 0:
        raise ValueError("Node-local inference requires positive replica and DP sizes")
    if expert_parallel_size != data_parallel_size:
        raise ValueError("inference_engine_node_local currently requires matching EP and DP sizes")
    if gpus_per_node is not None and data_parallel_size > gpus_per_node:
        raise ValueError(
            f"Node-local inference replica needs {data_parallel_size} GPUs but a node provides {gpus_per_node}"
        )


def validate_node_local_config(config: Mapping[str, Any], *, gpus_per_node: int | None = None) -> None:
    generator = config["generator"]
    if not isinstance(generator, Mapping):
        raise ValueError("generator configuration must be a mapping")
    if not generator.get("inference_engine_node_local", False):
        return
    validate_node_local_inference(
        enabled=generator.get("inference_engine_node_local", False),
        backend=generator["backend"],
        async_engine=generator["async_engine"],
        colocate_all=config["trainer"]["placement"]["colocate_all"],
        tensor_parallel_size=generator["inference_engine_tensor_parallel_size"],
        pipeline_parallel_size=generator["inference_engine_pipeline_parallel_size"],
        data_parallel_size=generator["inference_engine_data_parallel_size"],
        expert_parallel_size=generator["inference_engine_expert_parallel_size"],
        num_inference_engines=generator["num_inference_engines"],
        remote=not generator["run_engines_locally"],
        gpus_per_node=gpus_per_node,
    )


@dataclass(frozen=True)
class InferenceWorkerPlacement:
    host: str
    gpu_uuid: str
    dp_rank: int
    dp_world_size: int
    ep_rank: int
    ep_world_size: int
    torch_rank: int
    torch_world_size: int


@dataclass(frozen=True)
class InferenceReplicaPlacement:
    replica: int
    node_id: str
    bundle_index: int
    worker: InferenceWorkerPlacement
    expected_weight_receiver_rank: int


def validate_inference_replica_topology(
    placements: Sequence[InferenceReplicaPlacement],
    *,
    num_replicas: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    node_hosts: Mapping[str, str],
) -> None:
    """Verify actual workers and placement bundles before training can begin."""
    total = num_replicas * data_parallel_size
    if len(placements) != total:
        raise ValueError(f"Expected {total} inference workers, observed {len(placements)}")
    if {row.replica for row in placements} != set(range(num_replicas)):
        raise ValueError("Inference replica indices are incomplete")
    if len({row.worker.gpu_uuid for row in placements}) != total or any(not row.worker.gpu_uuid for row in placements):
        raise ValueError("Inference workers must have distinct, nonempty physical GPU UUIDs")
    receivers = [row.expected_weight_receiver_rank for row in placements]
    if len(set(receivers)) != total or set(receivers) != set(range(1, total + 1)):
        raise ValueError("Inference weight receiver ranks must be unique and cover the broadcast group")
    expected_ep_size = data_parallel_size if expert_parallel_size > 1 else 1
    for replica in range(num_replicas):
        rows = [row for row in placements if row.replica == replica]
        if len(rows) != data_parallel_size or {row.bundle_index for row in rows} != set(range(data_parallel_size)):
            raise ValueError(f"Inference replica {replica} has incomplete placement bundles")
        if len({row.node_id for row in rows}) != 1 or len({row.worker.host for row in rows}) != 1:
            raise ValueError(f"Inference replica {replica} spans nodes")
        for row in rows:
            worker = row.worker
            if node_hosts.get(row.node_id) != worker.host:
                raise ValueError(f"Inference replica {replica} worker host disagrees with its placement node")
            if worker.dp_rank != row.bundle_index or worker.torch_rank != row.bundle_index:
                raise ValueError(f"Inference replica {replica} worker ranks disagree with its placement bundle")
            if worker.dp_world_size != data_parallel_size or worker.torch_world_size != data_parallel_size:
                raise ValueError(f"Inference replica {replica} has an unexpected DP/torch world size")
            expected_ep_rank = row.bundle_index if expert_parallel_size > 1 else 0
            if worker.ep_world_size != expected_ep_size or worker.ep_rank != expected_ep_rank:
                raise ValueError(f"Inference replica {replica} has an unexpected EP rank or world size")
            if row.expected_weight_receiver_rank != 1 + replica * data_parallel_size + worker.torch_rank:
                raise ValueError(f"Inference replica {replica} has an incorrect weight receiver rank")
