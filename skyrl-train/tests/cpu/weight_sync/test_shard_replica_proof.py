"""Full original-byte comparisons over actual standalone CPU process groups."""

from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import ray
import torch
import torch.distributed as dist

from skyrl_train.weight_sync.shard_group_factory import GroupEndpoint, prepare_rank_groups
from skyrl_train.weight_sync.shard_group_schedule import (
    TrainerRank,
    ReceiverRank,
    ExpertEntry,
    build_shard_group_schedule,
)
from skyrl_train.weight_sync.shard_replica_proof import (
    ReplicaTensor,
    ReplicaCatalogue,
    build_replica_plan,
    FullReplicaComparator,
    borrowed_policy_groups,
)


def fixture_plan():
    trainers = tuple(TrainerRank(dp * 2 + ep, dp, 0, ep) for dp in range(2) for ep in range(2))
    receivers = tuple(ReceiverRank(ep, 0, ep) for ep in range(2))
    entries = tuple(
        ExpertEntry(f"expert{ep}.{projection}", 0, 0, ep, projection, 32)
        for ep in range(2)
        for projection in ("fc1", "fc2")
    )
    schedule = build_shard_group_schedule(
        trainers, receivers, entries, trainer_ep=2, receiver_ep=2, layers_by_pp=((0,),), num_experts=2
    )
    catalogue = tuple(
        ReplicaCatalogue(
            row.rank,
            (
                ReplicaTensor(f"expert{row.ep}", (139,), "bfloat16", True),
                ReplicaTensor("dense", (4,), "float32", False),
            ),
        )
        for row in trainers
    )
    return trainers, schedule, catalogue


class ReplicaActor:
    def __init__(self, rank, plan, directory, borrowed=False):
        self.rank, self.plan = rank, plan
        self.borrowed = borrowed
        ep = rank % 2
        self.sources = {
            f"expert{ep}": torch.arange(139, dtype=torch.bfloat16) + ep,
            "dense": torch.tensor([0, -2147483648, 2143289345, 1065353216], dtype=torch.int32).view(torch.float32),
        }
        self.original = {name: value.view(torch.uint8).clone() for name, value in self.sources.items()}
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/default-world" if borrowed else f"file://{directory}/default-{rank}",
            rank=rank if borrowed else 0,
            world_size=4 if borrowed else 1,
            timeout=timedelta(seconds=20),
        )
        self.endpoints = tuple(
            GroupEndpoint(item.name, item.members, "gloo", f"file://{directory}/{item.name}", 20)
            for item in plan.groups
        )

    def prepare(self):
        source_ranks = None
        if self.borrowed:
            self.groups = {}
            for item in self.plan.groups:
                group = dist.new_group(list(item.members), backend="gloo", timeout=timedelta(seconds=20))
                if self.rank in item.members:
                    self.groups[item.name] = group
            primary = next(item.name for item in self.plan.groups if item.primary and self.rank in item.members)
            dense = next(item.name for item in self.plan.groups if not item.primary and self.rank in item.members)
            parallel = SimpleNamespace(
                get_expert_data_parallel_group=lambda: self.groups[primary],
                get_data_parallel_group=lambda: self.groups[dense],
            )
            borrowed, source_ranks, receipts = borrowed_policy_groups(parallel, self.rank, fixture_plan()[1], self.plan)
            assert borrowed == self.groups and all(row["borrowed"] for row in receipts)
            wrong_primary = SimpleNamespace(
                get_expert_data_parallel_group=lambda: self.groups[dense],
                get_data_parallel_group=lambda: self.groups[dense],
            )
            with pytest.raises(ValueError, match="exact source-replica membership"):
                borrowed_policy_groups(wrong_primary, self.rank, fixture_plan()[1], self.plan)
        else:
            self.groups = prepare_rank_groups(self.rank, self.endpoints)
        self.comparator = FullReplicaComparator(
            self.rank,
            self.plan,
            self.groups,
            torch.empty(64, dtype=torch.uint8),
            torch.empty(17, dtype=torch.bool),
            broadcast_source_ranks=source_ranks,
        )
        return {name: (group.rank(), group.size()) for name, group in self.groups.items()}

    def compare(self, fault):
        for name, value in self.sources.items():
            value.view(torch.uint8).copy_(self.original[name])
        if fault == "expert" and self.rank == 2:
            self.sources["expert0"].view(torch.uint8)[3] ^= 1
        if fault == "dense" and self.rank in (1, 3):
            # Both DP copies in EP1 agree, so only the cross-EP dense proof bites.
            self.sources["dense"].view(torch.uint8)[3] ^= 128
        before = {name: value.view(torch.uint8).clone() for name, value in self.sources.items()}
        native_broadcast = dist.broadcast
        broadcast_sizes = []

        def measured_broadcast(tensor, *args, **kwargs):
            broadcast_sizes.append(tensor.numel())
            return native_broadcast(tensor, *args, **kwargs)

        with patch.object(dist, "broadcast", side_effect=measured_broadcast):
            result = self.comparator(self.sources, "manifest", 7, self.rank)
        assert all(torch.equal(value.view(torch.uint8), before[name]) for name, value in self.sources.items())
        return result, {**self.comparator.last_receipt, "observed_broadcast_sizes": broadcast_sizes}

    def close(self):
        for group in reversed(tuple(self.groups.values())):
            dist.destroy_process_group(group)
        dist.destroy_process_group()


@pytest.fixture(scope="module", params=[False, True])
def actors(tmp_path_factory, request):
    plan = build_replica_plan(*fixture_plan())
    directory = tmp_path_factory.mktemp("replica-proof")
    ray.init(num_cpus=4, include_dashboard=False)
    actor_class = ray.remote(num_cpus=1)(ReplicaActor)
    values = [actor_class.remote(rank, plan, str(directory), request.param) for rank in range(4)]
    ray.get([actor.prepare.remote() for actor in values], timeout=60)
    yield values
    ray.get([actor.close.remote() for actor in values], timeout=30)
    for actor in values:
        ray.kill(actor, no_restart=True)
    ray.shutdown()


@pytest.mark.parametrize("fault", ["none", "expert", "dense"])
def test_actual_custom_groups_compare_every_source_bit_without_mutation(actors, fault):
    results = ray.get([actor.compare.remote(fault) for actor in actors], timeout=60)
    assert all(proof.compared_bytes == proof.expected_bytes == 294 for proof, _ in results)
    assert all(len(receipt["groups"]) == 2 for _, receipt in results)
    # 278 expert bytes use five 64-byte transfers; dense 16-byte storage is
    # compared once in each group. The 17-byte bool scratch only chunks compares.
    assert all(sorted(receipt["observed_broadcast_sizes"]) == [16, 16, 22, 64, 64, 64, 64] for _, receipt in results)
    if fault == "none":
        assert all(proof.mismatches == 0 for proof, _ in results)
    elif fault == "expert":
        assert results[0][0].mismatches == results[2][0].mismatches == 1
        assert results[1][0].mismatches == results[3][0].mismatches == 0
    else:
        assert all(proof.mismatches == 2 for proof, _ in results)
        assert all(receipt["groups"][0]["mismatches"] == 0 for _, receipt in results)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "shape", "dense_ep"])
def test_catalogue_mismatch_fails_before_group_creation(fault):
    trainers, schedule, catalogue = fixture_plan()
    if fault == "missing":
        catalogue = catalogue[:-1]
    elif fault == "duplicate":
        catalogue = (*catalogue, catalogue[0])
    else:
        index = 2 if fault == "shape" else 1
        tensors = list(catalogue[index].tensors)
        target = 0 if fault == "shape" else 1
        tensors[target] = replace(tensors[target], shape=(8,))
        catalogue = tuple(
            replace(row, tensors=tuple(tensors)) if i == index else row for i, row in enumerate(catalogue)
        )
    with pytest.raises(ValueError):
        build_replica_plan(trainers, schedule, catalogue)
