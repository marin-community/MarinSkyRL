from collections import Counter

from ray.util.placement_group import PlacementGroup, placement_group_table

from skyrl_train.utils import get_reordered_bundle_indices


def validate_colocated_engine_geometry(*, tensor_pipeline_size: int, gpus_per_node: int) -> None:
    """Reject colocated engine shapes that cannot tile one policy node."""

    if tensor_pipeline_size > gpus_per_node:
        raise ValueError(
            f"A colocated inference engine requiring {tensor_pipeline_size} GPUs cannot fit on one "
            f"{gpus_per_node}-GPU policy node"
        )
    if gpus_per_node % tensor_pipeline_size != 0:
        raise ValueError(
            f"A colocated inference engine requiring {tensor_pipeline_size} GPUs does not divide a "
            f"{gpus_per_node}-GPU policy node into node-atomic engine slices"
        )


def colocated_engine_bundle_indices(
    *,
    reordered_bundle_indices: list[int],
    engine_index: int,
    data_parallel_rank: int,
    tensor_pipeline_size: int,
    data_parallel_size: int,
    gpus_per_node: int,
) -> list[int]:
    """Select a node-atomic TP/PP slice from node-ordered one-GPU bundles."""

    validate_colocated_engine_geometry(tensor_pipeline_size=tensor_pipeline_size, gpus_per_node=gpus_per_node)
    engines_per_node = gpus_per_node // tensor_pipeline_size
    replica_index = engine_index * data_parallel_size + data_parallel_rank
    node_index = replica_index // engines_per_node
    replica_within_node = replica_index % engines_per_node
    start = node_index * gpus_per_node + replica_within_node * tensor_pipeline_size
    stop = start + tensor_pipeline_size
    selected = reordered_bundle_indices[start:stop]
    if len(selected) != tensor_pipeline_size:
        raise ValueError(
            f"Colocated engine replica {replica_index} requires bundle offsets [{start}, {stop}), "
            f"but only {len(reordered_bundle_indices)} bundles are available"
        )
    return selected


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
