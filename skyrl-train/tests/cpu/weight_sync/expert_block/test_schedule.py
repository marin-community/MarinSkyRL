"""The expert-block schedule: every slot receives each of its transfers exactly once, from a holder, with no scratch copies."""

from collections import Counter
from dataclasses import replace
from itertools import product

import pytest

from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    ReceiverRank,
    Region,
    Schedule,
    TrainerRank,
    build_schedule,
    from_wire,
    receiver_participant,
    to_wire,
)

LAYERS_BY_PP = ((0, 1), (2,))
NUM_EXPERTS = 4
EP = 2
MATRIX_BYTES = 24
DENSE_NUMEL = {"model.norm.weight": 8, "model.layers.0.mlp.router.weight": 12, "model.layers.2.mlp.router.weight": 12}


def trainers(dp_count=2):
    ranks = []
    for dp, pp, ep in product(range(dp_count), range(len(LAYERS_BY_PP)), range(EP)):
        ranks.append(TrainerRank(len(ranks), dp, pp, ep))
    return tuple(ranks)


def receivers(replicas=2):
    return tuple(ReceiverRank(replica * EP + ep, replica, ep) for replica in range(replicas) for ep in range(EP))


def entries_of(trainer, shards=1):
    """The expert entries a trainer rank holds: its stage's layers, its EP block, every shard."""
    per_block = NUM_EXPERTS // EP
    result = []
    for layer in LAYERS_BY_PP[trainer.pp]:
        for expert in range(trainer.ep * per_block, (trainer.ep + 1) * per_block):
            for projection in ("fc1", "fc2"):
                for shard in range(shards):
                    suffix = f".shard{shard}" if shards > 1 else ""
                    name = f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}{suffix}"
                    result.append(
                        ExpertEntry(name, layer, trainer.pp, expert, projection, MATRIX_BYTES // shards, shard, shards)
                    )
    return result


def dense_of(trainer):
    """Every trainer rank of a stage holds the same dense runs (TP=1)."""
    if trainer.pp == 0:
        return [DenseSlice("model.layers.0.mlp.router.weight", 0, 12, "bfloat16", "router", 0, 0)]
    return [
        DenseSlice("model.norm.weight", 0, 8, "bfloat16", "norm", 0, 1),
        DenseSlice("model.layers.2.mlp.router.weight", 0, 12, "bfloat16", "router", 0, 1),
    ]


def build(**changes) -> Schedule:
    rows = trainers()
    arguments = dict(
        trainers=rows,
        receivers=receivers(),
        expert_inventories={row.rank: entries_of(row) for row in rows},
        dense_inventories={row.rank: dense_of(row) for row in rows},
        receiver_ep=EP,
        num_experts=NUM_EXPERTS,
        receiver_layers_by_pp=((0, 1, 2),),
        dense_holders={name: (0,) for name in DENSE_NUMEL},
        dense_numel=DENSE_NUMEL,
    )
    arguments.update(changes)
    return build_schedule(**arguments)


def test_every_receiver_slot_gets_each_of_its_expert_matrices_exactly_once():
    schedule = build()
    per_block = NUM_EXPERTS // EP
    for receiver in receivers():
        participant = receiver_participant(schedule.trainer_count, receiver)
        landed = Counter(item.entry.name for item in schedule.experts if participant in item.destinations)
        expected = {
            f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}"
            for layer in range(3)
            for expert in range(receiver.ep * per_block, (receiver.ep + 1) * per_block)
            for projection in ("fc1", "fc2")
        }
        assert set(landed) == expected
        assert set(landed.values()) == {1}
    assert dict(schedule.receiver_experts) == {
        receiver_participant(schedule.trainer_count, row): 3 * per_block * 2 for row in receivers()
    }


def test_each_broadcast_comes_from_a_holder_and_the_root_is_fixed_per_stage_and_block():
    schedule = build()
    holders = {row.rank: {entry.name for entry in entries_of(row)} for row in trainers()}
    by_rank = {row.rank: row for row in trainers()}
    roots = {}
    for item in schedule.experts:
        assert item.entry.name in holders[item.root]
        owner = by_rank[item.root]
        assert roots.setdefault((owner.pp, owner.ep), item.root) == item.root
    # Both data-parallel replicas take turns as root, so neither carries everything.
    assert {by_rank[root].dp for root in roots.values()} == {0, 1}


def test_groups_hold_one_root_and_only_the_receivers_of_its_block_so_nothing_lands_in_scratch():
    schedule = build()
    expert_groups = [group for group in schedule.groups if group.name.startswith("expert-")]
    assert len(expert_groups) == len(LAYERS_BY_PP) * EP
    for group in expert_groups:
        root, *members = group.members
        assert root < schedule.trainer_count
        block = int(group.name.rsplit("-", 1)[1])
        assert members == [receiver_participant(schedule.trainer_count, row) for row in receivers() if row.ep == block]
    # Trainers not chosen as a root belong to no group at all.
    roots = {group.members[0] for group in expert_groups}
    idle = set(range(schedule.trainer_count)) - roots
    assert idle and all(rank not in group.members for rank in idle for group in schedule.groups)


def test_receiver_ingress_is_its_expert_share_plus_every_dense_byte():
    schedule = build()
    expert_bytes = 3 * (NUM_EXPERTS // EP) * 2 * MATRIX_BYTES
    dense_bytes = sum(DENSE_NUMEL.values()) * 2
    assert dict(schedule.receiver_bytes) == {
        receiver_participant(schedule.trainer_count, row): expert_bytes + dense_bytes for row in receivers()
    }


def test_dense_slices_travel_in_the_expert_roots_groups_and_fan_out_per_replica():
    schedule = build()
    expert_group_names = {item.group for item in schedule.experts}
    assert [item.source.hf_name for item in schedule.dense] == sorted(DENSE_NUMEL)
    for item in schedule.dense:
        assert item.group in expert_group_names
        assert item.root in {row.rank for row in trainers() if row.pp == item.source.pp}
        assert len(item.landings) == 2
        assert item.local_groups == ("local-0-0", "local-1-0")


@pytest.mark.parametrize(
    "change,error",
    [
        ({"expert_inventories": {row.rank: entries_of(row)[1:] for row in trainers()}}, "exactly once"),
        ({"receivers": receivers()[:-1]}, "receiver topology"),
        ({"receiver_layers_by_pp": ((0, 1, 3),)}, "layer ownership"),
        ({"num_experts": 5}, "do not split evenly"),
        ({"dense_numel": {**DENSE_NUMEL, "model.norm.weight": 9}}, "cover 8 of 9"),
        ({"dense_holders": {name: (0,) for name in list(DENSE_NUMEL)[:2]}}, "Dense weights differ"),
    ],
)
def test_incomplete_or_inconsistent_inputs_are_refused(change, error):
    with pytest.raises(ValueError, match=error):
        build(**change)


def test_a_missing_inventory_is_refused():
    inventories = {row.rank: entries_of(row) for row in trainers()}
    inventories.pop(3)
    with pytest.raises(ValueError, match="must report its inventory"):
        build(expert_inventories=inventories)


def test_schedule_survives_the_json_round_trip_used_by_the_worker_rpc():
    schedule = build()
    assert from_wire(Schedule, to_wire(schedule)) == schedule


# --- Unequal expert-parallel degrees ---------------------------------------------


def test_a_finer_receiver_split_sends_each_owner_only_to_the_receivers_holding_its_experts():
    wide = tuple(ReceiverRank(replica * 4 + ep, replica, ep) for replica in range(2) for ep in range(4))
    schedule = build(receivers=wide, receiver_ep=4)
    for item in schedule.experts:
        block = item.entry.expert
        assert item.group == f"expert-{item.root}-0-{block}"
        assert item.destinations == tuple(
            receiver_participant(schedule.trainer_count, row) for row in wide if row.ep == block
        )
    expert_groups = [group for group in schedule.groups if group.name.startswith("expert-")]
    assert all(len(group.members) == 3 for group in expert_groups)
    assert {count for _, count in schedule.receiver_experts} == {3 * 2}


def test_a_coarser_receiver_split_gathers_two_owners_into_each_receiver():
    coarse = tuple(ReceiverRank(replica, replica, 0) for replica in range(2))
    schedule = build(receivers=coarse, receiver_ep=1)
    assert {item.group for item in schedule.experts} == {f"expert-{item.root}-0-0" for item in schedule.experts}
    assert len({item.root for item in schedule.experts}) == len(LAYERS_BY_PP) * EP
    assert {count for _, count in schedule.receiver_experts} == {3 * NUM_EXPERTS * 2}


# --- Receiver pipeline stages ---------------------------------------------------


def staged_receivers():
    return tuple(
        ReceiverRank((replica * 2 + stage) * EP + ep, replica, ep, stage)
        for replica in range(2)
        for stage in range(2)
        for ep in range(EP)
    )


STAGED_HOLDERS = {
    "model.layers.0.mlp.router.weight": (0,),
    "model.layers.2.mlp.router.weight": (1,),
    "model.norm.weight": (1,),
}


def test_receiver_stages_get_only_their_layers_and_the_dense_tensors_they_hold():
    staged = staged_receivers()
    schedule = build(receivers=staged, receiver_layers_by_pp=((0,), (1, 2)), dense_holders=STAGED_HOLDERS)
    stage_of = {receiver_participant(schedule.trainer_count, row): row.pp for row in staged}
    for item in schedule.experts:
        stage = 0 if item.entry.layer == 0 else 1
        assert item.group == f"expert-{item.root}-{stage}-{item.entry.expert // 2}"
        assert all(stage_of[d] == stage for d in item.destinations)
    assert {item.source.hf_name: item.local_groups for item in schedule.dense} == {
        "model.layers.0.mlp.router.weight": ("local-0-0", "local-1-0"),
        "model.layers.2.mlp.router.weight": ("local-0-1", "local-1-1"),
        "model.norm.weight": ("local-0-1", "local-1-1"),
    }
    experts = dict(schedule.receiver_experts)
    assert {experts[p] for p, stage in stage_of.items() if stage == 0} == {1 * 2 * 2}
    assert {experts[p] for p, stage in stage_of.items() if stage == 1} == {2 * 2 * 2}


def test_receiver_stages_must_hold_exactly_the_trainers_layers():
    with pytest.raises(ValueError, match="exactly the trainer's layers"):
        build(receivers=staged_receivers(), receiver_layers_by_pp=((0,), (1,)), dense_holders=STAGED_HOLDERS)


def test_a_dense_tensor_held_by_two_stages_is_sent_to_both():
    staged = tuple(ReceiverRank(stage * EP + ep, 0, ep, stage) for stage in range(2) for ep in range(EP))
    holders = {name: (0, 1) for name in DENSE_NUMEL}
    schedule = build(receivers=staged, receiver_layers_by_pp=((0,), (1, 2)), dense_holders=holders)
    assert len(schedule.dense) == 2 * len(DENSE_NUMEL)
    assert {tuple(item.local_groups) for item in schedule.dense} == {("local-0-0",), ("local-0-1",)}


# --- Trainer tensor parallelism ----------------------------------------------------


def tp_trainers():
    """Two TP ranks per (pp, ep) slot, DP=1: still eight ranks."""
    return tuple(
        TrainerRank(index, 0, pp, ep, tp) for index, (pp, ep, tp) in enumerate(product(range(2), range(EP), range(2)))
    )


def test_tensor_parallel_dense_shards_each_travel_once_and_tile_the_tensor():
    rows = tp_trainers()
    dense = {}
    for row in rows:
        # Column shard of the norm (one run of 4); row shard of a [4, 3] router (a column block: 4 runs of... 3/2).
        # Keep the router [2, 6] here so the block is 2 runs of 3 with stride 6.
        router = DenseSlice("model.layers.2.mlp.router.weight", row.tp * 3, 6, "bfloat16", "router", 0, 1, 2, 6)
        if row.pp == 0:
            dense[row.rank] = [
                DenseSlice("model.layers.0.mlp.router.weight", row.tp * 3, 6, "bfloat16", "router", 0, 0, 2, 6)
            ]
        else:
            dense[row.rank] = [DenseSlice("model.norm.weight", row.tp * 4, 4, "bfloat16", "norm", 0, 1), router]
    schedule = build(
        trainers=rows, expert_inventories={row.rank: entries_of(row) for row in rows}, dense_inventories=dense
    )
    transfers = Counter((item.source.hf_name, item.source.hf_offset) for item in schedule.dense)
    assert set(transfers.values()) == {1}
    assert sorted(transfers) == [
        ("model.layers.0.mlp.router.weight", 0),
        ("model.layers.0.mlp.router.weight", 3),
        ("model.layers.2.mlp.router.weight", 0),
        ("model.layers.2.mlp.router.weight", 3),
        ("model.norm.weight", 0),
        ("model.norm.weight", 4),
    ]
    by_rank = {row.rank: row for row in rows}
    for item in schedule.dense:
        # A shard can only come from a rank of its tensor-parallel index.
        assert by_rank[item.root].tp == item.source.hf_offset // (3 if item.source.runs == 2 else 4)
    assert Region(3, 6, 2, 6).intervals() == [(3, 6), (9, 12)]


def test_expert_tensor_parallel_shards_are_scheduled_per_shard_from_the_rank_holding_them():
    rows = tp_trainers()
    experts = {row.rank: [entry for entry in entries_of(row, shards=2) if entry.shard == row.tp] for row in rows}
    schedule = build(
        trainers=rows, expert_inventories=experts, dense_inventories={row.rank: dense_of(row) for row in rows}
    )
    assert {count for _, count in schedule.receiver_experts} == {3 * 2 * 2 * 2}
    by_rank = {row.rank: row for row in rows}
    assert all(by_rank[item.root].tp == item.entry.shard for item in schedule.experts)


def test_a_shard_count_disagreement_is_refused():
    experts = {row.rank: entries_of(row) for row in trainers()}
    experts[0] = [replace(experts[0][0], shards=2)] + experts[0][1:]
    with pytest.raises(ValueError, match="disagree on expert transfer|different shard counts"):
        build(expert_inventories=experts)
