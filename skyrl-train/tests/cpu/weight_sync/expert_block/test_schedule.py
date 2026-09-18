"""The expert-block schedule: every slot receives its block exactly once, from its owner, with no scratch copies."""

from collections import Counter
from dataclasses import replace
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


def trainers(dp_count=2):
    ranks = []
    for dp, pp, ep in product(range(dp_count), range(len(LAYERS_BY_PP)), range(EP)):
        ranks.append(TrainerRank(len(ranks), dp, pp, ep))
    return tuple(ranks)


def receivers(replicas=2):
    return tuple(ReceiverRank(replica * EP + ep, replica, ep) for replica in range(replicas) for ep in range(EP))


def entries():
    owner = {layer: pp for pp, layers in enumerate(LAYERS_BY_PP) for layer in layers}
    return tuple(
        ExpertEntry(
            f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}",
            layer,
            owner[layer],
            expert,
            projection,
            MATRIX_BYTES,
        )
        for layer in owner
        for expert in range(NUM_EXPERTS)
        for projection in ("fc1", "fc2")
    )


def dense():
    return (
        DenseSlice("model.norm.weight", 0, 8, "bfloat16", "decoder.final_layernorm.weight", 0, 1),
        DenseSlice("model.layers.0.mlp.router.weight", 0, 12, "bfloat16", "decoder.layers.0.mlp.router.weight", 0, 0),
        DenseSlice("model.layers.2.mlp.router.weight", 0, 12, "bfloat16", "decoder.layers.0.mlp.router.weight", 0, 1),
    )


def build(**changes) -> Schedule:
    arguments = dict(
        trainers=trainers(),
        receivers=receivers(),
        entries=entries(),
        dense=dense(),
        trainer_ep=EP,
        receiver_ep=EP,
        layers_by_pp=LAYERS_BY_PP,
        num_experts=NUM_EXPERTS,
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


def test_each_broadcast_comes_from_the_trainer_owning_that_stage_and_block():
    schedule = build()
    by_rank = {row.rank: row for row in trainers()}
    for item in schedule.experts:
        owner = by_rank[item.root]
        assert (owner.pp, owner.ep) == (item.entry.pp, item.entry.expert // (NUM_EXPERTS // EP))
        assert item.group == f"expert-{owner.pp}-0-{owner.ep}-{owner.ep}"


def test_groups_hold_one_root_and_only_the_receivers_of_that_block_so_nothing_lands_in_scratch():
    schedule = build()
    expert_groups = [group for group in schedule.groups if group.name.startswith("expert-")]
    assert len(expert_groups) == len(LAYERS_BY_PP) * EP
    for group in expert_groups:
        root, *members = group.members
        assert root < schedule.trainer_count
        assert all(member >= schedule.trainer_count for member in members)
        ep = int(group.name.rsplit("-", 1)[1])
        assert members == [receiver_participant(schedule.trainer_count, row) for row in receivers() if row.ep == ep]
        # With equal degrees every group pairs a block with itself.
        assert group.name.split("-")[3] == group.name.split("-")[4]
    # Trainers not chosen as a root belong to no group at all.
    roots = {group.members[0] for group in expert_groups}
    idle = set(range(schedule.trainer_count)) - roots
    assert idle and all(not schedule.groups_of(rank) for rank in idle)


def test_receiver_ingress_is_its_expert_share_plus_every_dense_byte():
    schedule = build()
    expert_bytes = len([item for item in entries() if item.expert < NUM_EXPERTS // EP]) * MATRIX_BYTES
    dense_bytes = sum(item.nbytes for item in dense())
    assert dict(schedule.receiver_bytes) == {
        receiver_participant(schedule.trainer_count, row): expert_bytes + dense_bytes for row in receivers()
    }
    # A receiver never lands more than its 1/EP share of the expert bytes.
    assert expert_bytes == len(entries()) * MATRIX_BYTES // EP


def test_dense_slices_split_across_the_stage_owners_and_fan_out_per_replica():
    schedule = build()
    assert [item.source.hf_name for item in schedule.dense] == sorted(item.hf_name for item in dense())
    for item in schedule.dense:
        assert item.group.startswith(f"expert-{item.source.pp}-0-")
        assert len(item.landings) == 2
        assert item.local_groups == ("local-0-0", "local-1-0")
    # Sorted by name, the slices rotate through the EP owners of their own stage.
    assert [item.group for item in schedule.dense] == ["expert-0-0-0-0", "expert-1-0-1-1", "expert-1-0-0-0"]


# --- Unequal expert-parallel degrees ---------------------------------------------


def test_a_finer_receiver_split_sends_each_owner_only_to_the_receivers_holding_its_experts():
    wide = tuple(ReceiverRank(replica * 4 + ep, replica, ep) for replica in range(2) for ep in range(4))
    schedule = build(receivers=wide, receiver_ep=4)
    # Trainer block 0 owns experts 0-1; receiver blocks 0 and 1 serve one each.
    for item in schedule.experts:
        trainer_block, receiver_block = item.entry.expert // 2, item.entry.expert
        assert item.group == f"expert-{item.entry.pp}-0-{trainer_block}-{receiver_block}"
        assert item.destinations == tuple(
            receiver_participant(schedule.trainer_count, row) for row in wide if row.ep == receiver_block
        )
    expert_groups = [group for group in schedule.groups if group.name.startswith("expert-")]
    assert len(expert_groups) == len(LAYERS_BY_PP) * 4
    assert all(len(group.members) == 3 for group in expert_groups)
    assert {count for _, count in schedule.receiver_experts} == {3 * 2}


def test_a_coarser_receiver_split_gathers_two_owners_into_each_receiver():
    coarse = tuple(ReceiverRank(replica, replica, 0) for replica in range(2))
    schedule = build(receivers=coarse, receiver_ep=1)
    assert {item.group for item in schedule.experts} == {
        f"expert-{pp}-0-{block}-0" for pp in range(2) for block in range(2)
    }
    assert {count for _, count in schedule.receiver_experts} == {3 * NUM_EXPERTS * 2}
    assert {root for root in (item.root for item in schedule.experts)} == {0, 1, 6, 7}


def test_expert_counts_must_split_evenly_on_both_sides():
    with pytest.raises(ValueError, match="do not split evenly"):
        build(receiver_ep=3)


# --- Receiver pipeline stages ---------------------------------------------------


def test_receiver_stages_get_only_their_layers_and_the_dense_tensors_they_hold():
    staged = tuple(
        ReceiverRank((replica * 2 + stage) * EP + ep, replica, ep, stage)
        for replica in range(2)
        for stage in range(2)
        for ep in range(EP)
    )
    holders = {
        "model.layers.0.mlp.router.weight": (0,),
        "model.layers.2.mlp.router.weight": (1,),
        "model.norm.weight": (1,),
    }
    schedule = build(receivers=staged, receiver_layers_by_pp=((0,), (1, 2)), dense_holders=holders)
    for item in schedule.experts:
        stage = 0 if item.entry.layer == 0 else 1
        assert item.group == f"expert-{item.entry.pp}-{stage}-{item.entry.expert // 2}-{item.entry.expert // 2}"
        assert all(
            next(row for row in staged if receiver_participant(schedule.trainer_count, row) == d).pp == stage
            for d in item.destinations
        )
    dense_stages = {item.source.hf_name: item.local_groups for item in schedule.dense}
    assert dense_stages == {
        "model.layers.0.mlp.router.weight": ("local-0-0", "local-1-0"),
        "model.layers.2.mlp.router.weight": ("local-0-1", "local-1-1"),
        "model.norm.weight": ("local-0-1", "local-1-1"),
    }
    stage0 = [receiver_participant(schedule.trainer_count, row) for row in staged if row.pp == 0]
    experts = dict(schedule.receiver_experts)
    assert {experts[p] for p in stage0} == {1 * 2 * 2}
    stage1 = [receiver_participant(schedule.trainer_count, row) for row in staged if row.pp == 1]
    assert {experts[p] for p in stage1} == {2 * 2 * 2}


def test_receiver_stages_must_hold_exactly_the_trainers_layers():
    staged = tuple(ReceiverRank(stage * EP + ep, 0, ep, stage) for stage in range(2) for ep in range(EP))
    with pytest.raises(ValueError, match="exactly the trainer's layers"):
        build(receivers=staged, receiver_layers_by_pp=((0,), (1,)))


def test_a_dense_tensor_held_by_two_stages_is_sent_to_both():
    staged = tuple(ReceiverRank(stage * EP + ep, 0, ep, stage) for stage in range(2) for ep in range(EP))
    holders = {
        name: (0, 1)
        for name in ("model.layers.0.mlp.router.weight", "model.layers.2.mlp.router.weight", "model.norm.weight")
    }
    schedule = build(receivers=staged, receiver_layers_by_pp=((0,), (1, 2)), dense_holders=holders)
    assert len(schedule.dense) == 2 * len(dense())
    assert {tuple(item.local_groups) for item in schedule.dense} == {("local-0-0",), ("local-0-1",)}


@pytest.mark.parametrize(
    "change,error",
    [
        ({"entries": entries()[1:]}, "do not cover"),
        ({"entries": entries() + (entries()[0],)}, "duplicate"),
        ({"entries": (replace(entries()[0], pp=1), *entries()[1:])}, "Unexpected"),
        ({"trainers": trainers()[1:]}, "trainer topology"),
        ({"receivers": receivers()[:-1]}, "receiver topology"),
        ({"layers_by_pp": ((0, 1), (3,))}, "layer ownership"),
        ({"num_experts": 5}, "do not split evenly"),
        ({"dense_holders": {"model.norm.weight": (0,)}}, "No receiver stage holds"),
        ({"dense": (replace(dense()[0], hf_offset=1),)}, "gap, overlap"),
    ],
)
def test_incomplete_or_inconsistent_inputs_are_refused(change, error):
    with pytest.raises(ValueError, match=error):
        build(**change)


def test_schedule_survives_the_json_round_trip_used_by_the_worker_rpc():
    schedule = build()
    assert from_wire(Schedule, to_wire(schedule)) == schedule
