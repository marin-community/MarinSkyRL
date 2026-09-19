"""Node-local inference replica placement: the rule that selects it and the verified placement records.

A node-local replica is one vLLM data-parallel group whose ranks all sit on one
physical node; with pipeline parallelism each stage's data-parallel group sits on
one node. ``auto`` selects it wherever it cannot place a run worse than the flat
group does; the engine factory then verifies the workers it got against the
bundles it was allocated before training starts. This module imports nothing from
the trainer, so the config validator can import it before any model module.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from marinskyrl.runtime_options import NodeLocalPlacement, WeightSyncTransport


def node_local_blocker(
    *,
    mode: NodeLocalPlacement,
    backend: str,
    async_engine: bool,
    colocated: bool,
    remote: bool,
    mp_executor: bool,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    data_parallel_size: int,
    expert_parallel_size: int,
    num_inference_engines: int,
    node_gpu_capacities: Sequence[int] | None = None,
) -> str | None:
    """Why an engine is not placed node-locally, or None when it is.

    ``node_gpu_capacities`` lists the GPUs of each live GPU node. It is known only once the Ray
    cluster is; the config check passes None and the engine factory asks again with it.
    """
    if mode is NodeLocalPlacement.OFF:
        return "generator.inference_engine_node_local is off"
    if backend != "vllm" or not async_engine or colocated or remote:
        return "it needs local, non-colocated async vLLM engines"
    if mp_executor:
        return "the mp executor places its own workers"
    if tensor_parallel_size != 1:
        return "it needs TP=1"
    if data_parallel_size < 2:
        return "a DP=1 engine has no replica to pack"
    if expert_parallel_size != data_parallel_size:
        return "it needs EP equal to DP"
    if node_gpu_capacities is None:
        return None
    node_size = max(node_gpu_capacities, default=0)
    if data_parallel_size > node_size:
        return f"a replica stage needs {data_parallel_size} GPUs but the largest node has {node_size}"
    stages = num_inference_engines * pipeline_parallel_size
    if sum(capacity // data_parallel_size for capacity in node_gpu_capacities) < stages:
        return f"the cluster's nodes cannot hold {stages} replica stages of {data_parallel_size} GPUs"
    engine_gpus = data_parallel_size * pipeline_parallel_size
    if mode is NodeLocalPlacement.AUTO and len(node_gpu_capacities) > 1 and engine_gpus % node_size:
        # One group per engine that fills part of a node can scatter across nodes and leave the
        # policy group without whole nodes (see use_per_engine_strict_pack_pg).
        return (
            f"an engine's {engine_gpus} GPUs are not a whole number of {node_size}-GPU nodes, so a group of "
            "its own could leave nodes partly used; generator.inference_engine_node_local=require packs it anyway"
        )
    return None


def validate_node_local_config(config: Mapping[str, Any]) -> None:
    """Refuse an unknown mode, and ``require`` for an engine shape that can never be node-local."""
    generator = config["generator"]
    mode = NodeLocalPlacement(generator.get("inference_engine_node_local", NodeLocalPlacement.AUTO))
    if mode is not NodeLocalPlacement.REQUIRE:
        return
    blocker = node_local_blocker(**node_local_engine_shape(config, mode))
    if blocker is not None:
        raise ValueError(f"generator.inference_engine_node_local=require cannot be honoured: {blocker}")


def node_local_engine_shape(config: Mapping[str, Any], mode: NodeLocalPlacement) -> dict[str, Any]:
    """The ``node_local_blocker`` arguments a resolved training configuration determines."""
    generator = config["generator"]
    tp_pp_size = (
        generator["inference_engine_tensor_parallel_size"] * generator["inference_engine_pipeline_parallel_size"]
    )
    return dict(
        mode=mode,
        backend=generator["backend"],
        async_engine=generator["async_engine"],
        colocated=config["trainer"]["placement"]["colocate_all"],
        remote=not generator["run_engines_locally"],
        mp_executor=bool(generator.get("inference_engine_mp_backend", False)) and tp_pp_size > 1,
        tensor_parallel_size=generator["inference_engine_tensor_parallel_size"],
        pipeline_parallel_size=generator["inference_engine_pipeline_parallel_size"],
        data_parallel_size=generator["inference_engine_data_parallel_size"],
        expert_parallel_size=generator["inference_engine_expert_parallel_size"],
        num_inference_engines=generator["num_inference_engines"],
    )


def validate_expert_block_transport(config: Mapping[str, Any]) -> None:
    """Refuse the expert-block weight-sync transport unless every precondition holds.

    The transport pairs each Megatron expert matrix with the vLLM worker that serves it and
    reads each HF tensor as one or more runs of one trainer parameter, so it needs the megatron
    strategy at TP=1 and ETP=1, local async vLLM engines at TP=1 placed node-locally, and the
    NCCL weight-sync backend.
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
    if generator["weight_sync_backend"] != "nccl":
        problems.append("generator.weight_sync_backend must be nccl")
    # The transport pairs trainer ranks with the verified placement of node-local replicas.
    mode = NodeLocalPlacement(generator["inference_engine_node_local"])
    blocker = node_local_blocker(**node_local_engine_shape(config, mode))
    if blocker is not None:
        problems.append(f"the engines must be placed node-locally, and {blocker}")
    if int(generator["expert_block_sync"]["timeout_seconds"]) <= 0:
        problems.append("generator.expert_block_sync.timeout_seconds must be positive")
    if problems:
        raise ValueError("generator.weight_sync_transport=expert_block requires: " + "; ".join(problems))


def validate_expert_block_trainer(config: Mapping[str, Any], *, uses_fully_async_trainer: bool) -> None:
    """Refuse the expert-block transport for an entrypoint that selects another trainer.

    Only ``FullyAsyncRayPPOTrainer`` runs it; any other trainer would sync by broadcast.
    """
    if (
        config["generator"]["weight_sync_transport"] == WeightSyncTransport.EXPERT_BLOCK
        and not uses_fully_async_trainer
    ):
        raise ValueError(
            "generator.weight_sync_transport=expert_block requires an entrypoint that runs "
            "FullyAsyncRayPPOTrainer (skyrl_train.entrypoints.fully_async, or terminal_bench without colocation)"
        )


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
    # A dense model has no expert-parallel group: vLLM reports size 1 on every worker whatever EP was asked for.
    expert_parallel = expert_parallel_size > 1 and any(row.worker.ep_world_size != 1 for row in placements)
    expected_ep_size = data_parallel_size if expert_parallel else 1
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
            expected_ep_rank = worker.dp_rank if expert_parallel else 0
            if worker.ep_world_size != expected_ep_size or worker.ep_rank != expected_ep_rank:
                raise ValueError(f"Inference replica {replica} has an unexpected EP rank or world size")
            if row.weight_receiver_rank != 1 + replica * per_replica + worker.torch_rank:
                raise ValueError(f"Inference replica {replica} has an incorrect weight receiver rank")
