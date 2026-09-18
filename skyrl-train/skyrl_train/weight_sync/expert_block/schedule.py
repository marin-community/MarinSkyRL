"""The collective schedule for expert-block weight sync, as a pure function of topology.

Trainer ranks are Megatron ranks; receiver ranks are vLLM workers, one per
expert-parallel slot per pipeline stage per inference replica. The trainer that
owns expert block ``k_t`` of its stage holds every expert matrix in that block,
and the receiver serving block ``k_r`` of its stage holds a live slot for each.
One NCCL group per (trainer stage, receiver stage, trainer block, receiver
block) joins that owner to those receivers across replicas, and every expert
matrix is one broadcast from its owner straight into the receiver's slot. With
one inference replica that is a point-to-point copy; with more, NCCL's tree
carries the fan-out.

Dense (non-expert) weights are split across the owners of their trainer stage,
sent to one receiver per replica on each receiver stage that holds the tensor,
and fanned out to that stage's other receivers over a node-local group.

Geometries beyond the qualified one (equal expert-parallel degree, one receiver
stage) are handled by the sections marked below: unequal expert-parallel
degrees pair every overlapping (trainer block, receiver block); receiver
pipeline stages route each layer, and each dense tensor, to the stage that
holds it.

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
    """Expert-owner coordinates; ``dp`` is Megatron's expert data-parallel rank."""

    rank: int
    dp: int
    pp: int
    ep: int


@dataclass(frozen=True)
class ReceiverRank:
    """A vLLM worker: ``pp`` is its pipeline stage, 0 when the engine has one stage."""

    rank: int
    replica: int
    ep: int
    pp: int = 0


@dataclass(frozen=True)
class ExpertEntry:
    """One expert matrix as the trainer exports it: ``fc1`` is ``[gate;up]``, ``fc2`` is ``down``."""

    name: str
    layer: int
    pp: int
    expert: int
    projection: str
    nbytes: int


@dataclass(frozen=True)
class DenseSlice:
    """A contiguous run of one dense HF tensor, backed by a contiguous run of one trainer parameter."""

    hf_name: str
    hf_offset: int
    numel: int
    wire_dtype: str
    source_key: str
    source_offset: int
    pp: int

    @property
    def nbytes(self) -> int:
        return self.numel * WIRE_DTYPE_BYTES[self.wire_dtype]


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
    # Per receiver participant: wire bytes it must land per sync, and expert matrices it must install.
    receiver_bytes: tuple[tuple[int, int], ...]
    receiver_experts: tuple[tuple[int, int], ...]

    def groups_of(self, participant: int) -> tuple[Group, ...]:
        return tuple(group for group in self.groups if participant in group.members)


def receiver_participant(trainer_count: int, receiver: ReceiverRank) -> int:
    return trainer_count + receiver.rank


def build_schedule(
    trainers: Sequence[TrainerRank],
    receivers: Sequence[ReceiverRank],
    entries: Sequence[ExpertEntry],
    dense: Sequence[DenseSlice],
    *,
    trainer_ep: int,
    receiver_ep: int,
    layers_by_pp: Sequence[Sequence[int]],
    num_experts: int,
    receiver_layers_by_pp: Sequence[Sequence[int]] | None = None,
    dense_holders: Mapping[str, Sequence[int]] | None = None,
) -> Schedule:
    """Order every collective identically for every participant, without touching tensors.

    ``receiver_layers_by_pp`` lists the layers each receiver stage holds (default: one
    stage with every layer); ``dense_holders`` maps each dense HF tensor to the receiver
    stages holding it (default: every stage).
    """
    if num_experts % trainer_ep or num_experts % receiver_ep:
        raise ValueError(f"{num_experts} experts do not split evenly across EP {trainer_ep} and EP {receiver_ep}")
    layer_count = sum(map(len, layers_by_pp))
    if receiver_layers_by_pp is None:
        receiver_layers_by_pp = (tuple(range(layer_count)),)
    dp_count = len({row.dp for row in trainers})
    pp_count = len(layers_by_pp)
    replica_count = len({row.replica for row in receivers})
    receiver_pp_count = len(receiver_layers_by_pp)
    trainer_at = {(row.dp, row.pp, row.ep): row for row in trainers}
    receiver_at = {(row.replica, row.pp, row.ep): row for row in receivers}
    if len(trainer_at) != len(trainers) or set(trainer_at) != set(
        product(range(dp_count), range(pp_count), range(trainer_ep))
    ):
        raise ValueError("Incomplete or duplicate trainer topology")
    if len(receiver_at) != len(receivers) or set(receiver_at) != set(
        product(range(replica_count), range(receiver_pp_count), range(receiver_ep))
    ):
        raise ValueError("Incomplete or duplicate receiver topology")
    layer_owner = _layer_stages(layers_by_pp, "trainer")
    receiver_stage = _layer_stages(receiver_layers_by_pp, "receiver")
    if set(receiver_stage) != set(layer_owner):
        raise ValueError("Receiver stages do not hold exactly the trainer's layers")
    trainer_count = len(trainers)
    if sorted(row.rank for row in trainers) != list(range(trainer_count)):
        raise ValueError("Trainer native ranks must be 0..N-1")
    if sorted(row.rank for row in receivers) != list(range(len(receivers))):
        raise ValueError("Receiver native ranks must be 0..N-1")
    if dense_holders is None:
        dense_holders = {item.hf_name: tuple(range(receiver_pp_count)) for item in dense}

    def root(pp: int, block: int) -> int:
        # Stages alternate their root across expert data-parallel replicas to spread egress.
        return trainer_at[pp % dp_count, pp, block].rank

    def receivers_of(stage: int, block: int) -> tuple[int, ...]:
        return tuple(
            receiver_participant(trainer_count, receiver_at[replica, stage, block]) for replica in range(replica_count)
        )

    # --- Unequal expert-parallel degrees ---------------------------------------------
    # Expert e is owned by trainer block e // per_trainer_block and served by receiver
    # block e // per_receiver_block. With equal degrees every group pairs block k with
    # block k; otherwise every overlapping pair of blocks gets its own group, so an
    # owner talks only to the receivers that hold some of its experts.
    per_trainer_block = num_experts // trainer_ep
    per_receiver_block = num_experts // receiver_ep
    groups: dict[str, Group] = {}

    def expert_group(trainer_stage: int, stage: int, trainer_block: int, receiver_block: int) -> str:
        name = f"expert-{trainer_stage}-{stage}-{trainer_block}-{receiver_block}"
        if name not in groups:
            groups[name] = Group(name, (root(trainer_stage, trainer_block), *receivers_of(stage, receiver_block)))
        return name

    expected = set(product(range(layer_count), range(num_experts), ("fc1", "fc2")))
    seen = set()
    experts = []
    for entry in sorted(entries, key=lambda item: item.name):
        key = (entry.layer, entry.expert, entry.projection)
        if key not in expected or key in seen or entry.pp != layer_owner[entry.layer] or entry.nbytes <= 0:
            raise ValueError(f"Unexpected or duplicate expert entry {entry.name}")
        seen.add(key)
        trainer_block, receiver_block = entry.expert // per_trainer_block, entry.expert // per_receiver_block
        stage = receiver_stage[entry.layer]
        name = expert_group(entry.pp, stage, trainer_block, receiver_block)
        experts.append(ExpertBroadcast(entry, name, root(entry.pp, trainer_block), receivers_of(stage, receiver_block)))
    if seen != expected:
        raise ValueError("Expert entries do not cover every layer, expert and projection")

    # --- Receiver pipeline stages -------------------------------------------------
    # A stage's receivers fan a dense tensor out among themselves on their own local
    # group, and a dense tensor goes to every stage that holds it (tied embeddings may
    # live on two stages).
    local_groups = {}
    for replica, stage in product(range(replica_count), range(receiver_pp_count)):
        name = f"local-{replica}-{stage}"
        local_groups[name] = Group(name, tuple(receivers_of(stage, block)[replica] for block in range(receiver_ep)))

    dense_broadcasts = []
    by_name: dict[str, list[DenseSlice]] = {}
    for item in dense:
        by_name.setdefault(item.hf_name, []).append(item)
    counter = 0
    for name in sorted(by_name):
        holders = tuple(dense_holders.get(name, ()))
        if not holders:
            raise ValueError(f"No receiver stage holds dense weight {name}")
        cursor = 0
        for item in sorted(by_name[name], key=lambda item: item.hf_offset):
            if item.hf_offset != cursor or item.numel <= 0 or item.wire_dtype not in WIRE_DTYPE_BYTES:
                raise ValueError(f"Dense slices of {name} have a gap, overlap or unsupported dtype")
            cursor += item.numel
            trainer_block = counter % trainer_ep
            counter += 1
            receiver_block = (trainer_block * per_trainer_block) // per_receiver_block
            for stage in holders:
                dense_broadcasts.append(
                    DenseBroadcast(
                        item,
                        expert_group(item.pp, stage, trainer_block, receiver_block),
                        root(item.pp, trainer_block),
                        receivers_of(stage, receiver_block),
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
