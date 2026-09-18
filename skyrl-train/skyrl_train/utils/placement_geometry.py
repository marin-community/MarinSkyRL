from collections.abc import Sequence
from enum import StrEnum


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


class EnginePlacementLayout(StrEnum):
    """How an inference engine's GPU bundles are indexed in its placement group."""

    # Colocated with the policy: node-ordered one-GPU bundles of the shared placement group.
    HYBRID = "hybrid"
    # One STRICT_PACK placement group per engine with engine-local bundle indices.
    PER_ENGINE = "per_engine"
    # One {GPU: tp*pp} bundle per (engine, DP rank) in a shared PACK placement group.
    MP = "mp"


def data_parallel_rank_bundle_indices(
    layout: EnginePlacementLayout,
    *,
    engine_index: int,
    data_parallel_size: int,
    tensor_pipeline_size: int,
    colocated_engine_bundles: Sequence[Sequence[int]] = (),
) -> list[int]:
    """Return the bundle index holding each DP rank's first GPU for one engine."""

    if layout is EnginePlacementLayout.HYBRID:
        return [
            colocated_engine_bundles[engine_index * data_parallel_size + rank][0] for rank in range(data_parallel_size)
        ]
    if layout is EnginePlacementLayout.PER_ENGINE:
        return [rank * tensor_pipeline_size for rank in range(data_parallel_size)]
    return [engine_index * data_parallel_size + rank for rank in range(data_parallel_size)]
