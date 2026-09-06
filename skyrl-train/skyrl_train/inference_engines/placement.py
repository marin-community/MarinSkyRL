from collections import Counter
from collections.abc import Mapping, Sequence
import socket

import torch

from ray.util.placement_group import PlacementGroup, placement_group_table

from skyrl_train.utils import get_reordered_bundle_indices
from skyrl_train.utils.placement_geometry import colocated_engine_bundle_indices
from marinskyrl.inference_placement import (
    InferenceReplicaPlacement,
    InferenceWorkerPlacement,
    validate_inference_replica_topology,
)


def inference_worker_placement(
    *, dp_rank: int, dp_world_size: int, ep_rank: int, ep_world_size: int
) -> InferenceWorkerPlacement:
    """Read physical identity inside the process that owns the inference device."""
    return InferenceWorkerPlacement(
        host=socket.gethostname(),
        gpu_uuid=str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid),
        dp_rank=dp_rank,
        dp_world_size=dp_world_size,
        ep_rank=ep_rank,
        ep_world_size=ep_world_size,
        torch_rank=torch.distributed.get_rank(),
        torch_world_size=torch.distributed.get_world_size(),
    )


def node_local_bundle_nodes(
    placement_groups: Sequence[PlacementGroup], *, data_parallel_size: int, node_gpu_capacities: Mapping[str, int]
) -> list[str]:
    """Check every GPU bundle is present and each replica fits on one Ray node."""
    node_ids = []
    for replica, pg in enumerate(placement_groups):
        bundles = placement_group_table(pg)["bundles_to_node_id"]
        if set(bundles) != set(range(data_parallel_size)):
            raise ValueError(f"Inference replica {replica} has incomplete placement bundles")
        if len(set(bundles.values())) != 1 or not bundles[0]:
            raise ValueError(f"Inference replica {replica} placement spans nodes")
        node_ids.append(bundles[0])
    for node_id, replica_count in Counter(node_ids).items():
        if replica_count * data_parallel_size > node_gpu_capacities.get(node_id, 0):
            raise ValueError(f"Inference replicas exceed GPU capacity on placement node {node_id}")
    return node_ids


def verified_inference_replica_placements(
    reports: Sequence[Sequence[Mapping[str, str | int]]],
    *,
    replica_nodes: Sequence[str],
    node_hosts: Mapping[str, str],
    relative_rank_offsets: Sequence[int],
    data_parallel_size: int,
    expert_parallel_size: int,
) -> list[InferenceReplicaPlacement]:
    """Join worker observations to their allocated bundles and verify the result."""
    if len(reports) != len(replica_nodes) * data_parallel_size or len(reports) != len(relative_rank_offsets):
        raise ValueError("Incomplete inference replica diagnostics")
    placements = []
    for index, (report, offset) in enumerate(zip(reports, relative_rank_offsets, strict=True)):
        if len(report) != 1:
            raise ValueError("Node-local TP=PP=1 requires one worker diagnostic per DP actor")
        worker = InferenceWorkerPlacement(**report[0])
        replica, dp_rank = divmod(index, data_parallel_size)
        placements.append(
            InferenceReplicaPlacement(
                replica=replica,
                node_id=replica_nodes[replica],
                bundle_index=dp_rank,
                worker=worker,
                expected_weight_receiver_rank=1 + offset + worker.torch_rank,
            )
        )
    validate_inference_replica_topology(
        placements,
        num_replicas=len(replica_nodes),
        data_parallel_size=data_parallel_size,
        expert_parallel_size=expert_parallel_size,
        node_hosts=node_hosts,
    )
    return placements


def colocated_engine_bundle_layout(
    shared_pg: PlacementGroup | None,
    *,
    num_inference_engines: int,
    data_parallel_size: int,
    tensor_pipeline_size: int,
) -> list[list[int]]:
    """Resolve node-atomic TP/PP slices, or return no slices when disaggregated."""

    if shared_pg is None:
        return []
    bundle_to_node_ids = placement_group_table(shared_pg)["bundles_to_node_id"]
    node_counts = Counter(bundle_to_node_ids.values())
    if not node_counts or len(set(node_counts.values())) != 1:
        raise ValueError(f"Colocated placement bundles must be uniform across nodes; got {dict(node_counts)}")
    gpus_per_node = next(iter(node_counts.values()))
    reordered_bundle_indices = get_reordered_bundle_indices(shared_pg)
    return [
        colocated_engine_bundle_indices(
            reordered_bundle_indices=reordered_bundle_indices,
            engine_index=engine_index,
            data_parallel_rank=data_parallel_rank,
            tensor_pipeline_size=tensor_pipeline_size,
            data_parallel_size=data_parallel_size,
            gpus_per_node=gpus_per_node,
        )
        for engine_index in range(num_inference_engines)
        for data_parallel_rank in range(data_parallel_size)
    ]
