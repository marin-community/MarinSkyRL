"""Node-local inference replica placement: the configuration rules and the verified placement records.

An opted-in replica is one vLLM data-parallel group whose ranks all sit on one
physical node; with pipeline parallelism each stage's data-parallel group sits
on one node. The trainer's config validation and the engine factory refuse a
configuration this cannot hold before any engine starts; the factory then
verifies the workers it actually got against the bundles it was allocated
before training starts. This module imports nothing from the trainer, so the
config validator can import it before any model module.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marinskyrl.runtime_options import WeightSyncTransport


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
    """Reject unsupported opt-ins; every other placement keeps its existing contract."""
    if not enabled:
        return
    if backend != "vllm" or not async_engine or colocate_all or remote:
        raise ValueError("inference_engine_node_local requires local, non-colocated async vLLM engines")
    if tensor_parallel_size != 1:
        raise ValueError("inference_engine_node_local requires TP=1")
    if data_parallel_size <= 0 or pipeline_parallel_size <= 0 or num_inference_engines <= 0:
        raise ValueError("inference_engine_node_local requires positive replica, PP and DP sizes")
    if expert_parallel_size != data_parallel_size:
        raise ValueError("inference_engine_node_local requires EP equal to DP")
    if gpus_per_node is not None and data_parallel_size > gpus_per_node:
        raise ValueError(
            f"Node-local inference replica needs {data_parallel_size} GPUs but a node provides {gpus_per_node}"
        )


def validate_node_local_config(config: Mapping[str, Any], *, gpus_per_node: int | None = None) -> None:
    """Validate the `generator` block of a resolved training configuration."""
    generator = config["generator"]
    if not generator.get("inference_engine_node_local", False):
        return
    validate_node_local_inference(
        enabled=True,
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


# Kept here rather than under weight_sync: the trainer's config validator imports this module
# before any model module, and the weight_sync package imports the models.
def validate_expert_block_transport(config: Mapping[str, Any]) -> None:
    """Refuse the expert-block weight-sync transport unless every precondition holds.

    The transport pairs each Megatron expert shard with the vLLM worker that serves it, so it
    needs the megatron strategy without tensor parallelism, local async vLLM engines at TP=1
    placed node-locally, and the NCCL weight-sync backend. The schedule can land tensor-parallel
    shards as regions of the receiver's tensors, but a TP2/ETP1 trainer showed expert replicas
    diverging across the TP pair after one update, so trainer TP stays refused.
    """
    generator = config["generator"]
    transport = generator["weight_sync_transport"]
    choices = [item.value for item in WeightSyncTransport]
    if transport not in choices:
        raise ValueError(f"generator.weight_sync_transport must be one of {choices}, not {transport!r}")
    if transport != WeightSyncTransport.EXPERT_BLOCK:
        return
    trainer = config["trainer"]
    problems = []
    if trainer["strategy"] != "megatron":
        problems.append("the policy must train with the megatron strategy")
    else:
        megatron = trainer["policy"]["megatron_config"]
        if megatron["tensor_model_parallel_size"] != 1:
            problems.append("the policy must use tensor_model_parallel_size 1")
        if megatron["expert_tensor_parallel_size"] not in (None, 1):
            problems.append("the policy must use expert_tensor_parallel_size 1")
        # Unequal expert-parallel degrees are paired by the schedule; each must divide the
        # expert count, which is checked against the model when the ranks report.
        if megatron["expert_model_parallel_size"] < 1 or generator["inference_engine_expert_parallel_size"] < 1:
            problems.append("expert-parallel sizes must be positive")
    if generator["backend"] != "vllm" or not generator["async_engine"] or not generator["run_engines_locally"]:
        problems.append("the engines must be local async vLLM engines")
    if trainer["placement"]["colocate_all"]:
        problems.append("the engines must not be colocated with the trainer")
    if generator["weight_sync_backend"] != "nccl":
        problems.append("generator.weight_sync_backend must be nccl")
    if not generator["inference_engine_node_local"]:
        problems.append("generator.inference_engine_node_local must be true")
    if generator["inference_engine_tensor_parallel_size"] != 1:
        problems.append("the engines must use TP=1")
    if int(generator["expert_block_sync"]["timeout_seconds"]) <= 0:
        problems.append("generator.expert_block_sync.timeout_seconds must be positive")
    if problems:
        raise ValueError("generator.weight_sync_transport=expert_block requires: " + "; ".join(problems))


@dataclass(frozen=True)
class InferenceWorkerPlacement:
    """What one vLLM worker observes about itself from inside its process."""

    host: str
    gpu_uuid: str
    dp_rank: int
    dp_world_size: int
    ep_rank: int
    ep_world_size: int
    torch_rank: int
    torch_world_size: int
    pp_rank: int = 0
    pp_world_size: int = 1


@dataclass(frozen=True)
class InferenceReplicaPlacement:
    """One worker joined to the replica and placement bundle it was allocated."""

    replica: int
    node_id: str
    bundle_index: int
    worker: InferenceWorkerPlacement
    weight_receiver_rank: int


def validate_inference_replica_topology(
    placements: Sequence[InferenceReplicaPlacement],
    *,
    num_replicas: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    pipeline_parallel_size: int = 1,
    node_hosts: Mapping[str, str],
) -> None:
    """Check the observed workers match their allocated bundles, each stage's DP group on one node.

    A worker's bundle is ``dp_rank * PP + pp_rank`` within its engine, which is also its torch
    rank in the engine's world of ``DP * PP`` ranks.
    """
    per_replica = data_parallel_size * pipeline_parallel_size
    total = num_replicas * per_replica
    if len(placements) != total:
        raise ValueError(f"Expected {total} inference workers, observed {len(placements)}")
    if {row.replica for row in placements} != set(range(num_replicas)):
        raise ValueError("Inference replica indices are incomplete")
    gpu_uuids = {row.worker.gpu_uuid for row in placements}
    if len(gpu_uuids) != total or "" in gpu_uuids:
        raise ValueError("Inference workers must have distinct, nonempty physical GPU UUIDs")
    if {row.weight_receiver_rank for row in placements} != set(range(1, total + 1)):
        raise ValueError("Inference weight receiver ranks must be unique and cover the broadcast group")
    expected_ep_size = data_parallel_size if expert_parallel_size > 1 else 1
    for replica in range(num_replicas):
        rows = [row for row in placements if row.replica == replica]
        if len(rows) != per_replica or {row.bundle_index for row in rows} != set(range(per_replica)):
            raise ValueError(f"Inference replica {replica} has incomplete placement bundles")
        for stage in range(pipeline_parallel_size):
            stage_rows = [row for row in rows if row.worker.pp_rank == stage]
            if len({row.node_id for row in stage_rows}) != 1 or len({row.worker.host for row in stage_rows}) != 1:
                raise ValueError(f"Inference replica {replica} stage {stage} spans nodes")
        for row in rows:
            worker = row.worker
            if node_hosts.get(row.node_id) != worker.host:
                raise ValueError(f"Inference replica {replica} worker host disagrees with its placement node")
            if worker.pp_world_size != pipeline_parallel_size or not 0 <= worker.pp_rank < pipeline_parallel_size:
                raise ValueError(f"Inference replica {replica} has an unexpected PP rank or world size")
            expected_rank = worker.dp_rank * pipeline_parallel_size + worker.pp_rank
            if row.bundle_index != expected_rank or worker.torch_rank != expected_rank:
                raise ValueError(f"Inference replica {replica} worker ranks disagree with its placement bundle")
            if worker.dp_world_size != data_parallel_size or worker.torch_world_size != per_replica:
                raise ValueError(f"Inference replica {replica} has an unexpected DP/torch world size")
            expected_ep_rank = worker.dp_rank if expert_parallel_size > 1 else 0
            if worker.ep_world_size != expected_ep_size or worker.ep_rank != expected_ep_rank:
                raise ValueError(f"Inference replica {replica} has an unexpected EP rank or world size")
            if row.weight_receiver_rank != 1 + replica * per_replica + worker.torch_rank:
                raise ValueError(f"Inference replica {replica} has an incorrect weight receiver rank")
