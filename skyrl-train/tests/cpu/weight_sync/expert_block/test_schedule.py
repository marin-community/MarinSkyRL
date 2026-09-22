"""The schedule sends each transfer once, from a rank that holds it, to the receivers that need it."""

from collections import Counter
from itertools import product

import pytest

from skyrl_train.weight_sync.expert_block.schedule import (
    DenseSlice,
    ExpertEntry,
    ReceiverRank,
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


def entries_of(trainer):
    """The expert entries of a trainer rank: the layers of its stage and the experts of its EP block."""
    per_block = NUM_EXPERTS // EP
    result = []
    for layer in LAYERS_BY_PP[trainer.pp]:
        for expert in range(trainer.ep * per_block, (trainer.ep + 1) * per_block):
            for projection in ("fc1", "fc2"):
                name = f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}"
                result.append(ExpertEntry(name, layer, trainer.pp, expert, projection, MATRIX_BYTES))
    return result


def dense_of(trainer):
    """Every trainer rank of a stage holds the same dense slices."""
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


def test_every_expert_matrix_is_sent_by_a_rank_holding_it_and_the_data_parallel_replicas_split_the_egress():
    schedule = build()
    holders = {row.rank: {entry.name for entry in entries_of(row)} for row in trainers()}
    replica_of = {row.rank: row.dp for row in trainers()}
    sent = Counter()
    for item in schedule.experts:
        assert item.entry.name in holders[item.root]
        sent[replica_of[item.root]] += item.entry.nbytes
    # Two data-parallel replicas hold every matrix; each sends half of the bytes.
    total = 3 * NUM_EXPERTS * 2 * MATRIX_BYTES
    assert sent == {0: total // 2, 1: total // 2}


def test_groups_hold_one_root_and_only_the_receivers_of_its_block():
    schedule = build()
    expert_groups = [group for group in schedule.groups if group.name.startswith("expert-")]
    assert len(expert_groups) == len(LAYERS_BY_PP) * EP
    for group in expert_groups:
        root, *members = group.members
        assert root < schedule.trainer_count
        block = int(group.name.rsplit("-", 1)[1])
        assert members == [receiver_participant(schedule.trainer_count, row) for row in receivers() if row.ep == block]
    # A trainer that is never a root belongs to no group.
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


def test_dense_slices_that_overlap_within_their_tensor_are_refused():
    inventories = {row.rank: dense_of(row) for row in trainers()}
    for rank, items in inventories.items():
        if items[0].hf_name == "model.norm.weight":
            # Two runs of the norm: [0, 5) and [4, 8) share element 4.
            inventories[rank] = [
                DenseSlice("model.norm.weight", 0, 5, "bfloat16", "norm", 0, 1),
                DenseSlice("model.norm.weight", 4, 4, "bfloat16", "norm", 4, 1),
                *items[1:],
            ]
    with pytest.raises(ValueError, match="gap or overlap at element 5"):
        build(dense_inventories=inventories)
