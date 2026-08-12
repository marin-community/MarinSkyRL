from collections import Counter

from ray.util.placement_group import PlacementGroup, placement_group_table

from skyrl_train.utils import get_reordered_bundle_indices
from skyrl_train.utils.placement_geometry import colocated_engine_bundle_indices


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
