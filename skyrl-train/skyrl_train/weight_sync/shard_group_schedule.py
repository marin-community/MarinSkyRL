"""Pure K10 expert-block collective schedule; logical bytes are not NIC traffic."""

from dataclasses import dataclass
from itertools import product


class UnequalExpertParallelism(ValueError):
    """K10 requires the same expert partition degree on both sides."""


@dataclass(frozen=True)
class TrainerRank:
    """Expert-owner coordinates: dp is Megatron expert_dp_rank, not dense dp_rank."""

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
    name: str
    layer: int
    pp: int
    expert: int
    projection: str
    nbytes: int


@dataclass(frozen=True)
class ExpertGroup:
    ep: int
    # Global custom-group identities: trainers first, then receivers.
    members: tuple[int, ...]


@dataclass(frozen=True)
class ExpertBroadcast:
    name: str
    group_ep: int
    root: int
    receiver_destinations: tuple[int, ...]
    nbytes: int


@dataclass(frozen=True)
class ShardGroupSchedule:
    groups: tuple[ExpertGroup, ...]
    broadcasts: tuple[ExpertBroadcast, ...]
    trainer_global_ranks: tuple[tuple[int, int], ...]
    receiver_global_ranks: tuple[tuple[int, int], ...]
    logical_root_bytes: tuple[tuple[int, int], ...]
    logical_receiver_bytes: tuple[tuple[int, int], ...]

    def collectives_for_member(self, global_rank: int) -> tuple[ExpertBroadcast, ...]:
        blocks = {group.ep for group in self.groups if global_rank in group.members}
        if not blocks:
            raise ValueError("Unknown collective member")
        return tuple(item for item in self.broadcasts if item.group_ep in blocks)


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def build_shard_group_schedule(
    trainers: tuple[TrainerRank, ...],
    receivers: tuple[ReceiverRank, ...],
    entries: tuple[ExpertEntry, ...],
    *,
    trainer_ep: int,
    receiver_ep: int,
    layers_by_pp: tuple[tuple[int, ...], ...],
    num_experts: int,
) -> ShardGroupSchedule:
    """Order identical collectives for every group member without allocating tensors.

    Input entries are the observed manifest's expert views, in its canonical order.
    Dense/shared weights use a separate schedule. A broadcast is issued once by
    every group member, including non-root trainers; recipient lists name final
    inference destinations, not a proposed NCCL routing tree or physical traffic.
    """
    if trainer_ep != receiver_ep:
        raise UnequalExpertParallelism(f"K10_EQUAL_EP_REQUIRED trainer={trainer_ep} receiver={receiver_ep}")
    _positive(trainer_ep, "expert parallelism")
    _positive(num_experts, "expert count")
    if num_experts % trainer_ep:
        raise ValueError("Expert count must divide evenly across EP blocks")
    if not trainers or not receivers or not layers_by_pp or not all(layers_by_pp):
        raise ValueError("Every topology and PP stage must be nonempty")
    for ranks in (trainers, receivers):
        if any(type(row.rank) is not int or row.rank < 0 for row in ranks):
            raise ValueError("Invalid native rank")
        if len({row.rank for row in ranks}) != len(ranks):
            raise ValueError("Duplicate native rank")
    dp_count = len({row.dp for row in trainers})
    replica_count = len({row.replica for row in receivers})
    pp_count = len(layers_by_pp)
    if any(type(value) is not int or value < 0 for row in trainers for value in (row.dp, row.pp, row.ep)) or any(
        type(value) is not int or value < 0 for row in receivers for value in (row.replica, row.ep)
    ):
        raise ValueError("Invalid topology coordinate")
    trainer_coordinates = {(row.dp, row.pp, row.ep): row for row in trainers}
    receiver_coordinates = {(row.replica, row.ep): row for row in receivers}
    if len(trainer_coordinates) != len(trainers) or set(trainer_coordinates) != set(
        product(range(dp_count), range(pp_count), range(trainer_ep))
    ):
        raise ValueError("Incomplete or duplicate trainer topology")
    if len(receiver_coordinates) != len(receivers) or set(receiver_coordinates) != set(
        product(range(replica_count), range(receiver_ep))
    ):
        raise ValueError("Incomplete or duplicate receiver topology")
    layer_owner = {layer: pp for pp, layers in enumerate(layers_by_pp) for layer in layers}
    if len(layer_owner) != sum(map(len, layers_by_pp)) or set(layer_owner) != set(range(len(layer_owner))):
        raise ValueError("Missing, duplicate or noncontiguous layer ownership")
    t_global = {row.rank: i for i, row in enumerate(sorted(trainers, key=lambda row: row.rank))}
    r_global = {row.rank: len(trainers) + i for i, row in enumerate(sorted(receivers, key=lambda row: row.rank))}
    groups = tuple(
        ExpertGroup(
            ep,
            tuple(t_global[trainer_coordinates[dp, pp, ep].rank] for dp in range(dp_count) for pp in range(pp_count))
            + tuple(r_global[receiver_coordinates[replica, ep].rank] for replica in range(replica_count)),
        )
        for ep in range(trainer_ep)
    )
    expected = set(product(range(len(layer_owner)), range(num_experts), ("fc1", "fc2")))
    observed = set()
    names = set()
    broadcasts = []
    root_bytes = dict.fromkeys(t_global.values(), 0)
    receiver_bytes = dict.fromkeys(r_global.values(), 0)
    for entry in entries:
        key = (entry.layer, entry.expert, entry.projection)
        if key not in expected or key in observed or not entry.name or entry.name in names:
            raise ValueError("Missing, duplicate or unknown expert manifest entry")
        if entry.pp != layer_owner[entry.layer]:
            raise ValueError("Manifest entry has the wrong PP owner")
        _positive(entry.nbytes, "manifest entry bytes")
        observed.add(key)
        names.add(entry.name)
        ep = entry.expert // (num_experts // trainer_ep)
        root_dp = entry.pp % dp_count
        root = t_global[trainer_coordinates[root_dp, entry.pp, ep].rank]
        destinations = tuple(r_global[receiver_coordinates[replica, ep].rank] for replica in range(replica_count))
        broadcasts.append(ExpertBroadcast(entry.name, ep, root, destinations, entry.nbytes))
        root_bytes[root] += entry.nbytes
        for destination in destinations:
            receiver_bytes[destination] += entry.nbytes
    if observed != expected:
        raise ValueError("Incomplete expert manifest coverage")
    return ShardGroupSchedule(
        groups,
        tuple(broadcasts),
        tuple(sorted(t_global.items())),
        tuple(sorted(r_global.items())),
        tuple(sorted(root_bytes.items())),
        tuple(sorted(receiver_bytes.items())),
    )
