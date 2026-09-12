"""CPU-only schedule semantics; no NIC-throughput inference from collective bytes."""

from dataclasses import replace
from itertools import product

import pytest

from skyrl_train.weight_sync.shard_group_schedule import (
    ExpertEntry,
    ReceiverRank,
    TrainerRank,
    UnequalExpertParallelism,
    build_shard_group_schedule,
)


def fixture(dp=2, pp=2, ep=8, replicas=1, layers=26, experts=256):
    trainers = tuple(
        TrainerRank(i, data, stage, expert)
        for i, (stage, data, expert) in enumerate(product(range(pp), range(dp), range(ep)))
    )
    receivers = tuple(ReceiverRank(i, *coordinate) for i, coordinate in enumerate(product(range(replicas), range(ep))))
    layer_groups = tuple(tuple(range(stage * layers // pp, (stage + 1) * layers // pp)) for stage in range(pp))
    entries = tuple(
        ExpertEntry(f"layer{layer}.expert{expert}.{projection}", layer, stage, expert, projection, nbytes)
        for stage, layer_ids in enumerate(layer_groups)
        for layer in layer_ids
        for expert in range(experts)
        for projection, nbytes in (("fc1", 2 * 1280 * 2560 * 2), ("fc2", 1280 * 2560 * 2))
    )
    return dict(
        trainers=trainers,
        receivers=receivers,
        entries=entries,
        trainer_ep=ep,
        receiver_ep=ep,
        layers_by_pp=layer_groups,
        num_experts=experts,
    )


@pytest.mark.parametrize("dp,replicas", [(2, 1), (2, 2), (1, 3)])
def test_required_snowball_matrix_all_members_roots_and_byte_coverage(dp, replicas):
    args = fixture(dp=dp, replicas=replicas)
    schedule = build_shard_group_schedule(**args)
    assert len(schedule.groups) == 8 and len(schedule.broadcasts) == 26 * 256 * 2
    for group in schedule.groups:
        assert group.members == tuple(
            r.rank for r in sorted(args["trainers"], key=lambda row: (row.dp, row.pp)) if r.ep == group.ep
        ) + tuple(len(args["trainers"]) + r.rank for r in args["receivers"] if r.ep == group.ep)
        assert len(group.members) == 2 * dp + replicas
        expected_calls = tuple(item for item in schedule.broadcasts if item.group_ep == group.ep)
        for member in group.members:
            assert schedule.collectives_for_member(member) == expected_calls
    roots = dict(schedule.logical_root_bytes)
    ingress = dict(schedule.logical_receiver_bytes)
    payload = sum(entry.nbytes for entry in args["entries"])
    assert sum(roots.values()) == payload
    assert set(ingress.values()) == {payload // 8}
    assert sum(ingress.values()) == replicas * payload
    for entry, broadcast in zip(args["entries"], schedule.broadcasts, strict=True):
        owner = next(r for r in args["trainers"] if r.rank == broadcast.root)
        assert owner.pp == entry.pp and owner.dp == entry.pp % dp and owner.ep == entry.expert // 32
        assert broadcast.root in schedule.groups[broadcast.group_ep].members
        assert broadcast.receiver_destinations == tuple(
            len(args["trainers"]) + r.rank for r in args["receivers"] if r.ep == owner.ep
        )
    # Every trainer DP replica is an eligible root for some PP stage at DP2.
    assert {r.dp for r in args["trainers"] if roots[r.rank]} == set(range(dp))


@pytest.mark.parametrize("replicas", [1, 2])
def test_tiny_two_sender_two_or_four_receiver_schedule(replicas):
    args = fixture(dp=1, pp=1, ep=2, replicas=replicas, layers=2, experts=4)
    result = build_shard_group_schedule(**args)
    assert len(result.trainer_global_ranks) == 2 and len(result.receiver_global_ranks) == 2 * replicas
    assert all(len(group.members) == 1 + replicas for group in result.groups)


def test_fewer_sending_replicas_do_not_multiply_logical_root_payload():
    one = build_shard_group_schedule(**fixture(dp=1, replicas=1))
    three = build_shard_group_schedule(**fixture(dp=1, replicas=3))
    assert one.logical_root_bytes == three.logical_root_bytes
    assert len(three.receiver_global_ranks) == 24
    # This is a collective-input invariant, not a claim about NIC egress.
    assert sum(v for _, v in three.logical_receiver_bytes) == 3 * sum(v for _, v in one.logical_receiver_bytes)


def test_named_unequal_ep_error_before_building_schedule():
    args = fixture()
    args["receiver_ep"] = 4
    with pytest.raises(UnequalExpertParallelism, match="K10_EQUAL_EP_REQUIRED"):
        build_shard_group_schedule(**args)


@pytest.mark.parametrize(
    "fault",
    ["missing_rank", "duplicate_rank", "missing_entry", "duplicate_entry", "wrong_pp", "wrong_expert", "bad_bytes"],
)
def test_invalid_native_topology_and_manifest_rejected(fault):
    args = fixture(dp=1, pp=2, ep=2, replicas=3, layers=2, experts=4)
    if fault == "missing_rank":
        args["trainers"] = args["trainers"][:-1]
    elif fault == "duplicate_rank":
        args["receivers"] += (args["receivers"][0],)
    elif fault == "missing_entry":
        args["entries"] = args["entries"][:-1]
    elif fault == "duplicate_entry":
        args["entries"] += (args["entries"][0],)
    else:
        replacement = {"wrong_pp": {"pp": 1}, "wrong_expert": {"expert": 4}, "bad_bytes": {"nbytes": True}}[fault]
        args["entries"] = (replace(args["entries"][0], **replacement), *args["entries"][1:])
    with pytest.raises(ValueError):
        build_shard_group_schedule(**args)


def test_schedule_is_independent_of_native_metadata_row_order():
    args = fixture(dp=1, pp=2, ep=2, replicas=3, layers=2, experts=4)
    expected = build_shard_group_schedule(**args)
    args["trainers"] = tuple(reversed(args["trainers"]))
    args["receivers"] = tuple(reversed(args["receivers"]))
    assert build_shard_group_schedule(**args) == expected
    with pytest.raises(ValueError, match="Unknown collective member"):
        expected.collectives_for_member(99)


def test_native_dense_dp_coordinates_are_not_expert_replica_coordinates():
    args = fixture()
    args["trainers"] = tuple(replace(row, dp=row.dp * 8 + row.ep) for row in args["trainers"])
    with pytest.raises(ValueError, match="trainer topology"):
        build_shard_group_schedule(**args)
