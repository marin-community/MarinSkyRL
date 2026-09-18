"""The collective schedule for expert-block weight sync, as a pure function of topology.

Trainer ranks are Megatron ranks with TP=1; receiver ranks are vLLM data-parallel
workers, one per expert-parallel slot per inference replica. Both sides must have
the same expert-parallel degree, so the trainer that owns expert block ``ep`` of
pipeline stage ``pp`` already holds exactly the experts every receiver with EP
rank ``ep`` serves. One NCCL group per (pp, ep) joins that trainer to those
receivers, and every expert matrix is one broadcast from its owner straight into
the receiver's live parameter slice. With one inference replica that is a
point-to-point copy; with more, NCCL's tree carries the fan-out.

Dense (non-expert) weights are split across the EP owners of their stage, sent
to one receiver per replica in the same groups, and fanned out to the replica's
other receivers over a node-local group.

Participants are numbered in one space: trainers by their native rank, then
receivers offset by the trainer count.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields, is_dataclass
from itertools import product
from typing import Any, get_args, get_origin

BF16 = "bfloat16"
FP32 = "float32"
WIRE_DTYPE_BYTES = {BF16: 2, FP32: 4}


class UnequalExpertParallelism(ValueError):
    """Expert-block sync requires the same expert-parallel degree on both sides."""


@dataclass(frozen=True)
class TrainerRank:
    """Expert-owner coordinates; ``dp`` is Megatron's expert data-parallel rank."""

    rank: int
    dp: int
    pp: int
    ep: int


@dataclass(frozen=True)
class ReceiverRank:
    rank: int
    replica: int
    ep: int


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
    # Wire bytes every receiver must land per sync, and expert matrices it must install.
    receiver_bytes: tuple[tuple[int, int], ...]
    receiver_expert_count: int

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
) -> Schedule:
    """Order every collective identically for every participant, without touching tensors."""
    if trainer_ep != receiver_ep:
        raise UnequalExpertParallelism(f"trainer EP {trainer_ep} differs from receiver EP {receiver_ep}")
    if num_experts % trainer_ep:
        raise ValueError("Expert count must divide evenly across EP blocks")
    dp_count = len({row.dp for row in trainers})
    pp_count = len(layers_by_pp)
    replica_count = len({row.replica for row in receivers})
    trainer_at = {(row.dp, row.pp, row.ep): row for row in trainers}
    receiver_at = {(row.replica, row.ep): row for row in receivers}
    if len(trainer_at) != len(trainers) or set(trainer_at) != set(
        product(range(dp_count), range(pp_count), range(trainer_ep))
    ):
        raise ValueError("Incomplete or duplicate trainer topology")
    if len(receiver_at) != len(receivers) or set(receiver_at) != set(product(range(replica_count), range(receiver_ep))):
        raise ValueError("Incomplete or duplicate receiver topology")
    layer_owner = {layer: pp for pp, layers in enumerate(layers_by_pp) for layer in layers}
    if sorted(layer_owner) != list(range(sum(map(len, layers_by_pp)))):
        raise ValueError("Missing, duplicate or noncontiguous layer ownership")
    trainer_count = len(trainers)
    if sorted(row.rank for row in trainers) != list(range(trainer_count)):
        raise ValueError("Trainer native ranks must be 0..N-1")
    if sorted(row.rank for row in receivers) != list(range(len(receivers))):
        raise ValueError("Receiver native ranks must be 0..N-1")

    def root(pp: int, ep: int) -> int:
        # Stages alternate their root across expert data-parallel replicas to spread egress.
        return trainer_at[pp % dp_count, pp, ep].rank

    def receivers_of(ep: int) -> tuple[int, ...]:
        return tuple(receiver_participant(trainer_count, receiver_at[replica, ep]) for replica in range(replica_count))

    groups = [
        Group(f"expert-{pp}-{ep}", (root(pp, ep), *receivers_of(ep)))
        for pp in range(pp_count)
        for ep in range(trainer_ep)
    ]
    local_groups = [
        Group(
            f"local-{replica}",
            tuple(receiver_participant(trainer_count, receiver_at[replica, ep]) for ep in range(receiver_ep)),
        )
        for replica in range(replica_count)
    ]

    expected = set(product(range(len(layer_owner)), range(num_experts), ("fc1", "fc2")))
    seen = set()
    experts = []
    per_block = num_experts // trainer_ep
    for entry in sorted(entries, key=lambda item: item.name):
        key = (entry.layer, entry.expert, entry.projection)
        if key not in expected or key in seen or entry.pp != layer_owner[entry.layer] or entry.nbytes <= 0:
            raise ValueError(f"Unexpected or duplicate expert entry {entry.name}")
        seen.add(key)
        ep = entry.expert // per_block
        experts.append(ExpertBroadcast(entry, f"expert-{entry.pp}-{ep}", root(entry.pp, ep), receivers_of(ep)))
    if seen != expected:
        raise ValueError("Expert entries do not cover every layer, expert and projection")

    dense_broadcasts = []
    by_name: dict[str, list[DenseSlice]] = {}
    for item in dense:
        by_name.setdefault(item.hf_name, []).append(item)
    counter = 0
    for name in sorted(by_name):
        cursor = 0
        for item in sorted(by_name[name], key=lambda item: item.hf_offset):
            if item.hf_offset != cursor or item.numel <= 0 or item.wire_dtype not in WIRE_DTYPE_BYTES:
                raise ValueError(f"Dense slices of {name} have a gap, overlap or unsupported dtype")
            cursor += item.numel
            ep = counter % trainer_ep
            counter += 1
            dense_broadcasts.append(
                DenseBroadcast(
                    item,
                    f"expert-{item.pp}-{ep}",
                    root(item.pp, ep),
                    receivers_of(ep),
                    tuple(group.name for group in local_groups),
                )
            )

    receiver_bytes = {receiver_participant(trainer_count, row): 0 for row in receivers}
    for broadcast in experts:
        for destination in broadcast.destinations:
            receiver_bytes[destination] += broadcast.entry.nbytes
    dense_bytes = sum(item.source.nbytes for item in dense_broadcasts)
    for participant in receiver_bytes:
        receiver_bytes[participant] += dense_bytes
    return Schedule(
        trainer_count,
        tuple(groups + local_groups),
        tuple(experts),
        tuple(dense_broadcasts),
        tuple(sorted(receiver_bytes.items())),
        len(experts) // trainer_ep,
    )


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
