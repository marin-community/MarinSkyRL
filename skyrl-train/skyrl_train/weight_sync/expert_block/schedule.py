"""The collective schedule for expert-block weight sync, as a pure function of what every rank holds.

Trainer ranks are Megatron ranks; receiver ranks are vLLM workers, one per
expert-parallel slot per pipeline stage per inference replica. Every trainer
rank reports the expert matrices (or shards of them) and the dense runs it
holds; every transfer is then one broadcast from one of its holders straight
into the receivers' live slots. With one inference replica that is a
point-to-point copy; with more, NCCL's tree carries the fan-out. One NCCL group
per (root, receiver stage, receiver block) joins a root to the receivers it
serves.

Dense (non-expert) weights are rotated across the expert roots of their stage
so they travel in the same groups, land on one receiver per replica on each
stage that holds the tensor, and fan out to that stage's other receivers over
a node-local group.

Geometries beyond the qualified one (TP=1 both sides, equal expert-parallel
degree, one receiver stage) are handled by the sections marked below:

* unequal expert-parallel degrees: an owner talks only to the receivers that
  hold some of its experts, because groups are keyed by receiver block;
* receiver pipeline stages: each layer, and each dense tensor, is routed to the
  stage that holds it;
* trainer tensor parallelism: a rank reports the shard it holds, described as
  a region of the full tensor; column shards are one run, row shards are a
  column block that the receiver lands through scratch.

Participants are numbered in one space: trainers by their native rank, then
receivers offset by the trainer count.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from itertools import product
from typing import Any, get_args, get_origin

BF16 = "bfloat16"
FP32 = "float32"
WIRE_DTYPE_BYTES = {BF16: 2, FP32: 4}


@dataclass(frozen=True)
class TrainerRank:
    """Megatron coordinates, informational: what a rank holds comes from its inventory."""

    rank: int
    dp: int
    pp: int
    ep: int
    tp: int = 0


@dataclass(frozen=True)
class ReceiverRank:
    """A vLLM worker: ``pp`` is its pipeline stage, 0 when the engine has one stage."""

    rank: int
    replica: int
    ep: int
    pp: int = 0


@dataclass(frozen=True)
class Region:
    """Where a contiguous payload lands in a flat tensor: ``runs`` runs of equal length, ``stride`` apart.

    One run is a plain ``narrow`` and a broadcast can land in it directly; a
    column block of a row-major matrix is ``rows`` runs and needs scratch.
    """

    offset: int
    numel: int
    runs: int = 1
    stride: int = 0

    @property
    def run_length(self) -> int:
        return self.numel // self.runs

    @property
    def direct(self) -> bool:
        return self.runs == 1

    def intervals(self) -> list[tuple[int, int]]:
        return [
            (self.offset + run * self.stride, self.offset + run * self.stride + self.run_length)
            for run in range(self.runs)
        ]


@dataclass(frozen=True)
class ExpertEntry:
    """One expert matrix, or one expert-tensor-parallel shard of it, as a trainer rank holds it.

    ``fc1`` is ``[gate;up]``, ``fc2`` is ``down``. With ``shards`` > 1 the rank holds shard
    ``shard``: rows of both halves for ``fc1``, a column block for ``fc2``.
    """

    name: str
    layer: int
    pp: int
    expert: int
    projection: str
    nbytes: int
    shard: int = 0
    shards: int = 1


@dataclass(frozen=True)
class DenseSlice:
    """A region of one dense HF tensor, backed by a contiguous run of one trainer parameter."""

    hf_name: str
    hf_offset: int
    numel: int
    wire_dtype: str
    source_key: str
    source_offset: int
    pp: int
    runs: int = 1
    stride: int = 0

    @property
    def nbytes(self) -> int:
        return self.numel * WIRE_DTYPE_BYTES[self.wire_dtype]

    @property
    def region(self) -> Region:
        return Region(self.hf_offset, self.numel, self.runs, self.stride)

    def identity(self) -> tuple:
        """What identifies the transfer independently of which rank backs it."""
        return (self.hf_name, self.hf_offset, self.numel, self.runs, self.stride, self.wire_dtype, self.pp)


@dataclass(frozen=True)
class Group:
    name: str
    members: tuple[int, ...]


@dataclass(frozen=True)
class ExpertBroadcast:
    entry: ExpertEntry
    group: str
    root: int
    destinations: tuple[int, ...]


@dataclass(frozen=True)
class DenseBroadcast:
    source: DenseSlice
    group: str
    root: int
    # One landing receiver per replica, each followed by a fan-out on its local group.
    landings: tuple[int, ...]
    local_groups: tuple[str, ...]


@dataclass(frozen=True)
class Schedule:
    trainer_count: int
    groups: tuple[Group, ...]
    experts: tuple[ExpertBroadcast, ...]
    dense: tuple[DenseBroadcast, ...]
    # Per receiver participant: wire bytes it must land per sync, and expert transfers it must install.
    receiver_bytes: tuple[tuple[int, int], ...]
    receiver_experts: tuple[tuple[int, int], ...]

    def groups_of(self, participant: int) -> tuple[Group, ...]:
        return tuple(group for group in self.groups if participant in group.members)


def receiver_participant(trainer_count: int, receiver: ReceiverRank) -> int:
    return trainer_count + receiver.rank


def build_schedule(
    trainers: Sequence[TrainerRank],
    receivers: Sequence[ReceiverRank],
    expert_inventories: Mapping[int, Sequence[ExpertEntry]],
    dense_inventories: Mapping[int, Sequence[DenseSlice]],
    *,
    receiver_ep: int,
    num_experts: int,
    receiver_layers_by_pp: Sequence[Sequence[int]],
    dense_holders: Mapping[str, Sequence[int]],
    dense_numel: Mapping[str, int],
) -> Schedule:
    """Order every collective identically for every participant, without touching tensors.

    ``expert_inventories`` and ``dense_inventories`` map each trainer rank to what it holds;
    ``receiver_layers_by_pp`` lists the layers each receiver stage holds; ``dense_holders``
    maps each dense HF tensor to the receiver stages holding it and ``dense_numel`` to its
    element count on the receiver.
    """
    trainer_count = len(trainers)
    if sorted(row.rank for row in trainers) != list(range(trainer_count)):
        raise ValueError("Trainer native ranks must be 0..N-1")
    if sorted(row.rank for row in receivers) != list(range(len(receivers))):
        raise ValueError("Receiver native ranks must be 0..N-1")
    if set(expert_inventories) != set(range(trainer_count)) or set(dense_inventories) != set(range(trainer_count)):
        raise ValueError("Every trainer rank must report its inventory")
    if num_experts % receiver_ep:
        raise ValueError(f"{num_experts} experts do not split evenly across receiver EP {receiver_ep}")
    replica_count = len({row.replica for row in receivers})
    receiver_pp_count = len(receiver_layers_by_pp)
    receiver_at = {(row.replica, row.pp, row.ep): row for row in receivers}
    if len(receiver_at) != len(receivers) or set(receiver_at) != set(
        product(range(replica_count), range(receiver_pp_count), range(receiver_ep))
    ):
        raise ValueError("Incomplete or duplicate receiver topology")
    receiver_stage = _layer_stages(receiver_layers_by_pp, "receiver")

    def receivers_of(stage: int, block: int) -> tuple[int, ...]:
        return tuple(
            receiver_participant(trainer_count, receiver_at[replica, stage, block]) for replica in range(replica_count)
        )

    groups: dict[str, Group] = {}

    def group_for(root: int, stage: int, block: int) -> str:
        name = f"expert-{root}-{stage}-{block}"
        if name not in groups:
            groups[name] = Group(name, (root, *receivers_of(stage, block)))
        return name

    # --- Expert transfers: every (layer, expert, projection, shard) exactly once, from one holder ---
    holders: dict[tuple, list[int]] = {}
    entries: dict[tuple, ExpertEntry] = {}
    for rank in sorted(expert_inventories):
        for entry in expert_inventories[rank]:
            key = (entry.layer, entry.expert, entry.projection, entry.shard)
            if key in entries and entries[key] != entry:
                raise ValueError(f"Trainer ranks disagree on expert transfer {entry.name}")
            entries.setdefault(key, entry)
            holders.setdefault(key, []).append(rank)
    shards = {entry.shards for entry in entries.values()}
    if len(shards) > 1:
        raise ValueError(f"Expert entries report different shard counts {sorted(shards)}")
    shard_count = shards.pop() if shards else 1
    layers = {entry.layer for entry in entries.values()}
    if layers != set(receiver_stage):
        raise ValueError("Receiver stages do not hold exactly the trainer's layers")
    expected = set(product(sorted(layers), range(num_experts), ("fc1", "fc2"), range(shard_count)))
    if set(entries) != expected:
        raise ValueError("Expert entries do not cover every layer, expert, projection and shard exactly once")
    per_receiver_block = num_experts // receiver_ep
    experts = []
    expert_roots: dict[tuple[int, int], set[int]] = {}
    for key in sorted(entries, key=lambda item: entries[item].name):
        entry = entries[key]
        # --- Unequal expert-parallel degrees: the group is keyed by the receiver block ---
        block = entry.expert // per_receiver_block
        stage = receiver_stage[entry.layer]
        ranks = holders[key]
        # Holders of one matrix are its data-parallel (and, for experts, tensor-parallel) replicas;
        # rotate the root by stage and block so a replica set shares its egress across them.
        root = ranks[(entry.pp + block) % len(ranks)]
        experts.append(ExpertBroadcast(entry, group_for(root, stage, block), root, receivers_of(stage, block)))
        expert_roots.setdefault((entry.pp, stage), set()).add(root)

    # --- Receiver pipeline stages: a stage's receivers fan a dense tensor out among themselves ---
    local_groups = {}
    for replica, stage in product(range(replica_count), range(receiver_pp_count)):
        name = f"local-{replica}-{stage}"
        local_groups[name] = Group(name, tuple(receivers_of(stage, block)[replica] for block in range(receiver_ep)))

    # --- Dense transfers: every region of every dense tensor exactly once, from a holder that already roots ---
    dense_holder_ranks: dict[tuple, list[int]] = {}
    slices: dict[tuple, DenseSlice] = {}
    for rank in sorted(dense_inventories):
        for item in dense_inventories[rank]:
            key = item.identity()
            slices.setdefault(key, item)
            dense_holder_ranks.setdefault(key, []).append(rank)
    by_name: dict[str, list[DenseSlice]] = {}
    for item in slices.values():
        by_name.setdefault(item.hf_name, []).append(item)
    if set(by_name) != set(dense_holders) or set(by_name) != set(dense_numel):
        raise ValueError(
            f"Dense weights differ: trainer only {sorted(set(by_name) - set(dense_holders))}, "
            f"receiver only {sorted(set(dense_holders) - set(by_name))}"
        )
    dense_broadcasts = []
    counter = 0
    for name in sorted(by_name):
        _check_regions_cover(name, [item.region for item in by_name[name]], dense_numel[name])
        for item in sorted(by_name[name], key=lambda item: (item.hf_offset, item.runs)):
            if item.wire_dtype not in WIRE_DTYPE_BYTES:
                raise ValueError(f"Dense slice of {name} has unsupported dtype {item.wire_dtype}")
            for stage in dense_holders[name]:
                candidates = sorted(
                    set(dense_holder_ranks[item.identity()]) & expert_roots.get((item.pp, stage), set())
                )
                if not candidates:
                    candidates = dense_holder_ranks[item.identity()]
                root = candidates[counter % len(candidates)]
                counter += 1
                # Reuse a group this root already has for the stage; otherwise it gets one for its own block.
                existing = [
                    group
                    for group in groups.values()
                    if group.members[0] == root and group.name.startswith(f"expert-{root}-{stage}-")
                ]
                if existing:
                    group_name = existing[0].name
                    block = int(group_name.rsplit("-", 1)[1])
                else:
                    block = root % receiver_ep
                    group_name = group_for(root, stage, block)
                dense_broadcasts.append(
                    DenseBroadcast(
                        item,
                        group_name,
                        root,
                        receivers_of(stage, block),
                        tuple(f"local-{replica}-{stage}" for replica in range(replica_count)),
                    )
                )

    receiver_bytes = {receiver_participant(trainer_count, row): 0 for row in receivers}
    receiver_experts = dict.fromkeys(receiver_bytes, 0)
    for broadcast in experts:
        for destination in broadcast.destinations:
            receiver_bytes[destination] += broadcast.entry.nbytes
            receiver_experts[destination] += 1
    for item in dense_broadcasts:
        stage = int(item.local_groups[0].rsplit("-", 1)[1])
        for block in range(receiver_ep):
            for participant in receivers_of(stage, block):
                receiver_bytes[participant] += item.source.nbytes
    return Schedule(
        trainer_count,
        tuple(groups.values()) + tuple(local_groups.values()),
        tuple(experts),
        tuple(dense_broadcasts),
        tuple(sorted(receiver_bytes.items())),
        tuple(sorted(receiver_experts.items())),
    )


def _layer_stages(layers_by_pp: Sequence[Sequence[int]], side: str) -> dict[int, int]:
    owner = {layer: pp for pp, layers in enumerate(layers_by_pp) for layer in layers}
    if sorted(owner) != list(range(sum(map(len, layers_by_pp)))):
        raise ValueError(f"Missing, duplicate or noncontiguous {side} layer ownership")
    return owner


def _check_regions_cover(name: str, regions: Sequence[Region], numel: int) -> None:
    """The regions must tile ``[0, numel)`` exactly once."""
    intervals = sorted(interval for region in regions for interval in region.intervals())
    cursor = 0
    for begin, end in intervals:
        if begin != cursor or end <= begin:
            raise ValueError(f"Dense slices of {name} have a gap or overlap at element {cursor}")
        cursor = end
    if cursor != numel:
        raise ValueError(f"Trainer slices cover {cursor} of {numel} elements of {name}")


def to_wire(value: Any) -> Any:
    """Plain JSON-compatible structure: dataclasses become dicts, tuples become lists."""
    if is_dataclass(value):
        return {field.name: to_wire(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, (tuple, list)):
        return [to_wire(item) for item in value]
    if isinstance(value, dict):
        return {key: to_wire(item) for key, item in value.items()}
    return value


def from_wire(kind: type, value: Any) -> Any:
    """Rebuild ``kind`` from :func:`to_wire` output, using the dataclass field annotations."""
    if is_dataclass(kind):
        hints = {field.name: field.type for field in fields(kind)}
        return kind(**{name: from_wire(hint, value[name]) for name, hint in hints.items()})
    origin = get_origin(kind)
    if origin is tuple:
        args = get_args(kind)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(from_wire(args[0], item) for item in value)
        return tuple(from_wire(arg, item) for arg, item in zip(args, value, strict=True))
    if not isinstance(value, kind):
        raise TypeError(f"Expected {kind.__name__}, got {type(value).__name__}")
    return value
