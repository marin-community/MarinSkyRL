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
    UnequalExpertParallelism,
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
    assert schedule.receiver_expert_count == 3 * per_block * 2


def test_each_broadcast_comes_from_the_trainer_owning_that_stage_and_block():
    schedule = build()
    by_rank = {row.rank: row for row in trainers()}
    for item in schedule.experts:
        owner = by_rank[item.root]
        assert (owner.pp, owner.ep) == (item.entry.pp, item.entry.expert // (NUM_EXPERTS // EP))
        assert item.group == f"expert-{owner.pp}-{owner.ep}"


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
        assert item.group == f"expert-{item.source.pp}-{item.group.rsplit('-', 1)[1]}"
        assert len(item.landings) == 2
        assert item.local_groups == ("local-0", "local-1")
    # Sorted by name, the slices rotate through the EP owners of their own stage.
    assert [item.group for item in schedule.dense] == ["expert-0-0", "expert-1-1", "expert-1-0"]


def test_unequal_expert_parallelism_is_refused_by_name():
    with pytest.raises(UnequalExpertParallelism, match="trainer EP 2 differs from receiver EP 4"):
        build(receiver_ep=4)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"entries": entries()[1:]}, "do not cover"),
        ({"entries": entries() + (entries()[0],)}, "duplicate"),
        ({"entries": (replace(entries()[0], pp=1), *entries()[1:])}, "Unexpected"),
        ({"trainers": trainers()[1:]}, "trainer topology"),
        ({"receivers": receivers()[:-1]}, "receiver topology"),
        ({"layers_by_pp": ((0, 1), (3,))}, "layer ownership"),
        ({"num_experts": 5}, "divide evenly"),
        ({"dense": (replace(dense()[0], hf_offset=1),)}, "gap, overlap"),
    ],
)
def test_incomplete_or_inconsistent_inputs_are_refused(change, error):
    with pytest.raises(ValueError, match=error):
        build(**change)


def test_schedule_survives_the_json_round_trip_used_by_the_worker_rpc():
    schedule = build()
    assert from_wire(Schedule, to_wire(schedule)) == schedule
