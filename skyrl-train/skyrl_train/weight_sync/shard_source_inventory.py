"""Exact per-expert source views for the TP1 Grug shard-group schedule.

Consumes resolved Bridge slice descriptors; creates no concatenated export or
communication buffers. Dense/shared descriptors remain in their original HF
order for a separate replicated stream. Native callers still own the weight
lease, rank catalogue, replica identity proof and stream completion.
"""

from dataclasses import dataclass
from itertools import product
import re

import torch

from skyrl_train.weight_sync.frozen_source_views import FrozenSourceSlice, source_view
from skyrl_train.weight_sync.shard_group_schedule import ExpertEntry, TrainerRank


EXPERT_NAME = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(gate|up|down)_proj\.weight")


@dataclass(frozen=True)
class LocalExpertSource:
    entry: ExpertEntry
    source_key: str
    shape: tuple[int, int]


@dataclass(frozen=True)
class ShardSourceInventory:
    experts: tuple[LocalExpertSource, ...]
    dense: tuple[FrozenSourceSlice, ...]


def local_shard_inventory(
    slices: tuple[FrozenSourceSlice, ...],
    sources: dict[str, torch.Tensor],
    trainer: TrainerRank,
    *,
    layers: tuple[int, ...],
    num_experts: int,
    expert_parallel_size: int,
    hidden_size: int,
    intermediate_size: int,
) -> ShardSourceInventory:
    """Prove every local expert matrix is one qualified contiguous BF16 view.

    Resolved global expert IDs and HF offsets must agree. A full gate/up pair
    must refer to adjacent halves of the same original parameter, exactly as
    GrugStackedGatedExpertMapping exports it. Missing, aliasing and foreign
    PP/EP views fail before any tensor is written.
    """
    if (
        expert_parallel_size <= 0
        or num_experts <= 0
        or num_experts % expert_parallel_size
        or not 0 <= trainer.ep < expert_parallel_size
        or not layers
        or len(set(layers)) != len(layers)
        or min(hidden_size, intermediate_size) <= 0
    ):
        raise ValueError("Invalid source inventory geometry")
    local_count = num_experts // expert_parallel_size
    first_expert = trainer.ep * local_count
    expected = set(product(layers, range(first_expert, first_expert + local_count), ("fc1", "fc2")))
    grouped = {}
    dense = []
    seen_slices = set()
    for item in slices:
        if item in seen_slices:
            raise ValueError("Duplicate resolved source slice")
        seen_slices.add(item)
        source_view(item, sources)  # Validate the current source storage and bounds.
        if not item.expert:
            dense.append(item)
            continue
        match = EXPERT_NAME.fullmatch(item.hf_name)
        if match is None or item.wire_dtype != "bfloat16":
            raise ValueError("Unqualified expert HF name or dtype")
        layer, projection = int(match[1]), match[2]
        matrix_size = hidden_size * intermediate_size
        if item.numel != matrix_size or item.hf_offset % matrix_size:
            raise ValueError("Expert slice geometry or global offset differs from model")
        expert = item.hf_offset // matrix_size
        key = (layer, expert, "fc2" if projection == "down" else "fc1")
        if key not in expected:
            raise ValueError("Expert source lies outside its observed PP/EP owner")
        parts = grouped.setdefault(key, {})
        if projection in parts:
            raise ValueError("Duplicate expert projection")
        parts[projection] = item
    if set(grouped) != expected:
        raise ValueError("Source inventory misses local expert matrices")
    experts = []
    ranges = []
    for (layer, expert, projection), parts in sorted(grouped.items()):
        needed = {"down"} if projection == "fc2" else {"gate", "up"}
        if set(parts) != needed or len({part.source_key for part in parts.values()}) != 1:
            raise ValueError("Gate/up source is incomplete or split across storage")
        first = parts["down" if projection == "fc2" else "gate"]
        if first.source_offset != 0 or (projection == "fc1" and parts["up"].source_offset != matrix_size):
            raise ValueError("Expert gate/up halves are reordered or nonadjacent")
        source = sources[first.source_key]
        shape = (hidden_size, intermediate_size) if projection == "fc2" else (2 * intermediate_size, hidden_size)
        if source.dtype != torch.bfloat16 or tuple(source.shape) != shape or not source.is_contiguous():
            raise ValueError("Expert source is not the exact contiguous BF16 matrix")
        begin, end = source.data_ptr(), source.data_ptr() + source.numel() * source.element_size()
        if any(source.device == device and max(begin, left) < min(end, right) for device, left, right in ranges):
            raise ValueError("Different expert matrices overlap source storage")
        ranges.append((source.device, begin, end))
        entry = ExpertEntry(
            f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}",
            layer,
            trainer.pp,
            expert,
            projection,
            source.numel() * source.element_size(),
        )
        experts.append(LocalExpertSource(entry, first.source_key, shape))
    return ShardSourceInventory(tuple(experts), tuple(dense))


def expert_source_view(item: LocalExpertSource, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    """Read current values through the original parameter, including later updates."""
    source = sources[item.source_key]
    if tuple(source.shape) != item.shape or source.dtype != torch.bfloat16 or not source.is_contiguous():
        raise ValueError("Expert storage geometry changed after preparation")
    return source.detach()


def expert_destination_view(item: LocalExpertSource, parameters, expert_maps, *, backend: str) -> torch.Tensor:
    """Resolve the complete installed matrix using the actual global expert map."""
    expected_shape = item.shape
    item = item.entry
    if backend != "TRITON" or item.projection not in ("fc1", "fc2"):
        raise ValueError("Direct expert views require the qualified TRITON projection")
    prefix = f"model.layers.{item.layer}.mlp.experts.routed_experts"
    mapping = expert_maps[prefix]
    owned = sorted(value for value in mapping if value >= 0)
    if any(type(value) is not int or value < -1 for value in mapping) or owned != list(range(len(owned))):
        raise ValueError("Expert map must uniquely cover local slots")
    if not 0 <= item.expert < len(mapping) or mapping[item.expert] < 0:
        raise ValueError("Receiver does not own the scheduled global expert")
    parameter = parameters[prefix + (".w13_weight" if item.projection == "fc1" else ".w2_weight")]
    if parameter.ndim != 3 or parameter.shape[0] != len(owned):
        raise ValueError("Receiver expert inventory differs from actual expert map")
    view = parameter[mapping[item.expert]]
    if (
        tuple(view.shape) != expected_shape
        or view.dtype != torch.bfloat16
        or not view.is_contiguous()
        or view.numel() * view.element_size() != item.nbytes
    ):
        raise ValueError("Receiver expert matrix differs from scheduled bytes or dtype")
    return view


@dataclass(frozen=True)
class DenseSourceRank:
    trainer: TrainerRank
    slices: tuple[FrozenSourceSlice, ...]


@dataclass(frozen=True)
class DenseTransfer:
    source: FrozenSourceSlice
    root_native_rank: int
    group_ep: int
    landing_native_ranks: tuple[int, ...]
    # Each landing fans out on its own inference replica's local group.
    replica_fanout: tuple[tuple[int, ...], ...]


def dense_stream_plan(rows, receivers, expected_shapes, *, expert_parallel_size):
    """Assign every dense HF element once, then one landing per inference replica.

    Metadata equality across trainer copies is necessary but is not a byte
    identity proof. The caller must verify those copies before allowing root
    rotation. Inter-node transfer uses the existing expert-block group; each
    receiver replica subsequently calls its own local broadcast in this order.
    Actual NIC traffic cannot be inferred from these logical destinations.
    """
    if not rows or not receivers or expert_parallel_size <= 0:
        raise ValueError("Dense stream needs complete trainer and receiver topology")
    if len({row.trainer.rank for row in rows}) != len(rows) or len({row.rank for row in receivers}) != len(receivers):
        raise ValueError("Dense stream native rank identity is duplicated")
    by_coord = {(row.trainer.dp, row.trainer.pp, row.trainer.ep): row for row in rows}
    dp_count = len({row.trainer.dp for row in rows})
    pp_count = len({row.trainer.pp for row in rows})
    if len(by_coord) != len(rows) or set(by_coord) != set(
        product(range(dp_count), range(pp_count), range(expert_parallel_size))
    ):
        raise ValueError("Dense stream trainer topology is incomplete")
    replica_count = len({row.replica for row in receivers})
    by_receiver = {(row.replica, row.ep): row for row in receivers}
    if len(by_receiver) != len(receivers) or set(by_receiver) != set(
        product(range(replica_count), range(expert_parallel_size))
    ):
        raise ValueError("Dense stream receiver topology is incomplete")
    canonical = []
    for pp in range(pp_count):
        reference = tuple(sorted(by_coord[0, pp, 0].slices, key=lambda item: (item.hf_name, item.hf_offset)))
        if any(item.expert for item in reference):
            raise ValueError("Expert matrices cannot enter the replicated stream")
        for dp, ep in product(range(dp_count), range(expert_parallel_size)):
            actual = tuple(sorted(by_coord[dp, pp, ep].slices, key=lambda item: (item.hf_name, item.hf_offset)))
            if actual != reference:
                raise ValueError("Trainer replicas disagree on dense source geometry")
        canonical.extend((pp, item) for item in reference)
    by_name = {}
    for pp, item in canonical:
        by_name.setdefault(item.hf_name, []).append((pp, item))
    if set(by_name) != set(expected_shapes):
        raise ValueError("Dense source names do not cover the installed manifest")
    plan = []
    for name, values in sorted(by_name.items()):
        shape, dtype = expected_shapes[name]
        expected_numel = 1
        for dimension in shape:
            if type(dimension) is not int or dimension <= 0:
                raise ValueError("Dense manifest shape is invalid")
            expected_numel *= dimension
        cursor = 0
        for pp, item in sorted(values, key=lambda pair: pair[1].hf_offset):
            if item.hf_offset != cursor or item.numel <= 0 or item.wire_dtype != dtype:
                raise ValueError("Dense stream has a coverage gap, overlap or dtype mismatch")
            cursor += item.numel
            ep = len(plan) % expert_parallel_size
            owner = by_coord[pp % dp_count, pp, ep].trainer.rank
            landings = tuple(by_receiver[replica, ep].rank for replica in range(replica_count))
            fanout = tuple(
                tuple(by_receiver[replica, local_ep].rank for local_ep in range(expert_parallel_size))
                for replica in range(replica_count)
            )
            plan.append(DenseTransfer(item, owner, ep, landings, fanout))
        if cursor != expected_numel:
            raise ValueError("Dense stream does not cover all installed elements")
    return tuple(plan)
