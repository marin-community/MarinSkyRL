"""Actual metadata/lease helpers; CPU tensors and explicit distributed metadata doubles."""

from dataclasses import replace
from itertools import product
from types import SimpleNamespace

import pytest
import torch

import skyrl_train.weight_sync.shard_preparation as preparation
from skyrl_train.weight_sync.policy_weight_access import PolicyWeightAccess
from skyrl_train.weight_sync.shard_group_factory import GroupEndpoint
from skyrl_train.weight_sync.shard_group_schedule import ReceiverRank, TrainerRank
from skyrl_train.weight_sync.shard_replica_proof import local_replica_catalogue
from skyrl_train.weight_sync.shard_session import ShardSession, storage_versions
from skyrl_train.weight_sync.shard_source_inventory import local_shard_inventory
from tests.cpu.weight_sync.test_frozen_source_views import task
from tests.cpu.weight_sync.test_shard_source_inventory import source_fixture
from tests.cpu.weight_sync.test_shard_stream import fixture_plan
from skyrl_train.weight_sync.shard_stream import ShardStreamRank


def metadata_fixture():
    geometry = preparation.ShardGeometry(8, 3, 2, ((0,), (1,)), 4, 3, 2)
    trainers = tuple(TrainerRank(i, dp, pp, ep) for i, (dp, pp, ep) in enumerate(product(range(2), range(2), range(2))))
    policy = []
    for trainer in trainers:
        slices, sources = source_fixture(trainer)
        policy.append(
            {
                "preparation_id": "fixture",
                "role": "policy",
                "rank": trainer.rank,
                "identity": {"physical_node": f"node-{trainer.rank}"},
                "trainer": trainer,
                "geometry": geometry,
                "inventory": local_shard_inventory(
                    slices,
                    sources,
                    trainer,
                    layers=(trainer.pp,),
                    num_experts=4,
                    expert_parallel_size=2,
                    hidden_size=3,
                    intermediate_size=2,
                ),
                "catalogue": local_replica_catalogue(trainer.rank, slices, sources),
                "native_groups": {
                    "expert": tuple(row.rank for row in trainers if (row.pp, row.ep) == (trainer.pp, trainer.ep)),
                    "dense": tuple(row.rank for row in trainers if row.pp == trainer.pp),
                },
            }
        )
    receivers = [
        {
            "preparation_id": "fixture",
            "role": "receiver",
            "rank": rank,
            "identity": {"physical_node": f"receiver-{rank}"},
            "receiver": ReceiverRank(rank, rank // 2, rank % 2),
            "geometry": geometry,
            "dense_parameters": {f"model.layers.{layer}.mlp.router.bias": ((4,), "float32") for layer in range(2)},
            "expected_bytes": 2 * (2 * (12 + 6) * 2 + 4 * 4),
        }
        for rank in range(6)
    ]
    return geometry, policy, receivers


def endpoint(name, members):
    return GroupEndpoint(name, members, "gloo", f"file:///explicit-fixture/{name}", 10)


def test_plan_joins_all_live_roles_and_preserves_receiver_byte_oracle():
    geometry, policy, receivers = metadata_fixture()
    options = preparation.PreparationOptions(1024, 64, 128, 0)
    plan = preparation.plan_shard_preparation(policy, receivers, geometry, options, endpoint)
    assert dict(plan.expected_receiver_bytes) == {rank: 176 for rank in range(8, 14)}
    assert len(plan.identity_rows) == 14
    assert len(plan.plan_id) == 64
    assert [group.name for group in plan.endpoints] == [
        "stream-expert-0",
        "stream-expert-1",
        "stream-local-0",
        "stream-local-1",
        "stream-local-2",
    ]
    assert all(not group.name.startswith("replica") for group in plan.endpoints)
    assert sum(count for _, count in plan.schedule.logical_receiver_bytes) == 3 * 2 * 4 * 36


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "foreign_role", "rank_map", "preparation", "group", "dense_storage"]
)
def test_plan_rejects_incomplete_or_foreign_native_metadata(fault):
    geometry, policy, receivers = metadata_fixture()
    if fault == "missing":
        policy.pop()
    elif fault == "duplicate":
        receivers[-1] = receivers[0]
    elif fault == "foreign_role":
        policy[0]["role"] = "receiver"
    elif fault == "rank_map":
        policy[0]["trainer"] = replace(policy[0]["trainer"], rank=7)
    elif fault == "preparation":
        receivers[0]["preparation_id"] = "stale"
    elif fault == "group":
        policy[0]["native_groups"]["expert"] = (0, 1)
    else:
        receivers[0]["dense_parameters"] = dict(receivers[0]["dense_parameters"], extra=((1,), "bfloat16"))
    with pytest.raises(ValueError):
        preparation.plan_shard_preparation(
            policy, receivers, geometry, preparation.PreparationOptions(1024, 64, 128, 0), endpoint
        )


def policy_worker(monkeypatch):
    geometry = preparation.ShardGeometry(8, 3, 2, ((0,), (1,)), 4, 3, 2)
    trainer = TrainerRank(0, 0, 0, 0)
    slices, sources = source_fixture(trainer)
    tasks = []
    for key, value in sources.items():
        rows = [row for row in slices if row.source_key == key]
        if "linear_fc1" in key:
            tasks.append(
                task("GrugStackedGatedExpertMapping", key, {"gate": rows[0].hf_name, "up": rows[1].hf_name}, value)
            )
        elif "linear_fc2" in key:
            tasks.append(task("GrugStackedExpertMapping", key, rows[0].hf_name, value))
        else:
            tasks.append(task("ReplicatedMapping", key, rows[0].hf_name, value))
    model = torch.nn.Linear(3, 2)
    worker = SimpleNamespace(
        _policy_weight_access=PolicyWeightAccess(),
        actor_module=[model],
        bridge=SimpleNamespace(get_conversion_tasks=lambda modules: tasks),
        provider=SimpleNamespace(tensor_model_parallel_size=1, num_moe_experts=4),
    )
    parallel = SimpleNamespace(
        get_expert_data_parallel_rank=lambda: 0,
        get_pipeline_model_parallel_rank=lambda: 0,
        get_expert_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_rank=lambda: 0,
        get_tensor_model_parallel_world_size=lambda: 1,
        get_expert_model_parallel_world_size=lambda: 2,
        get_expert_data_parallel_group=lambda: "expert",
        get_data_parallel_group=lambda: "dense",
    )
    monkeypatch.setattr(preparation.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(preparation.dist, "get_world_size", lambda: 8)
    monkeypatch.setattr(
        preparation.dist, "get_process_group_ranks", lambda group: (0, 4) if group == "expert" else (0, 1, 4, 5)
    )
    return worker, geometry, parallel


def test_live_bridge_metadata_holds_lease_until_explicit_cleanup(monkeypatch):
    worker, geometry, parallel = policy_worker(monkeypatch)
    receipts = []
    row = preparation.collect_policy_preparation(worker, "actual", geometry, parallel, {"rank": 0}, receipts.append)
    assert len(row["inventory"].experts) == 4
    assert row["source_setup_allocation"]["after"]["cuda_measured"] is False
    assert receipts[0]["phase"] == "source-metadata"
    with pytest.raises(RuntimeError, match="already owned"):
        worker._policy_weight_access.acquire("ppo")
    with pytest.raises(ValueError, match="different"):
        preparation.close_live_preparation(worker, "stale")
    preparation.close_live_preparation(worker, "actual")
    with worker._policy_weight_access.hold("ppo"):
        pass


def test_metadata_capture_failure_releases_actual_lease(monkeypatch):
    worker, geometry, parallel = policy_worker(monkeypatch)

    def failed_capture(row):
        raise OSError("durable sink failed")

    with pytest.raises(OSError, match="durable sink failed"):
        preparation.collect_policy_preparation(worker, "actual", geometry, parallel, {"rank": 0}, failed_capture)
    assert worker._policy_weight_access.owner is None
    assert not hasattr(worker, "_shard_preparation")


def session_fixture():
    trainers, receivers, views, schedule, dense, _ = fixture_plan(1, 1, 1)
    sources = {view.source_key: torch.ones(view.shape, dtype=torch.bfloat16) for view in views if view.entry.pp == 0}
    sources.update(
        {
            "layer0.qkv": torch.ones(6, dtype=torch.bfloat16),
            "layer0.router": torch.ones(2, dtype=torch.bfloat16),
            "layer0.bias": torch.ones(2),
        }
    )
    runner = ShardStreamRank(
        0,
        schedule,
        views,
        dense,
        sources,
        {},
        {},
        torch.empty(128, dtype=torch.uint8),
        {0: SimpleNamespace(size=lambda: 3, rank=lambda: 0)},
        None,
        dense_chunk_bytes=64,
    )
    access = PolicyWeightAccess()
    return ShardSession(runner, policy_access=access, replica_verifier=lambda *args: None, owned_groups=()), access


@pytest.mark.parametrize("begin", [False, True])
def test_preparation_lease_handoff_has_no_update_gap_and_closes(begin):
    session, access = session_fixture()
    token = access.acquire("preparation")
    session.adopt_preparation_lease(token, storage_versions(session.runner.sources))
    with pytest.raises(RuntimeError, match="already owned"):
        access.acquire("ppo")
    if begin:
        session.begin(session.manifest_id, 3)
    with pytest.raises(RuntimeError, match="already owned"):
        access.acquire("ppo")
    session.close(session.manifest_id, 3)
    with access.hold("ppo"):
        pass


def test_modified_source_cannot_cross_preparation_handoff():
    session, access = session_fixture()
    token = access.acquire("preparation")
    session.adopt_preparation_lease(token, storage_versions(session.runner.sources))
    next(iter(session.runner.sources.values())).add_(1)
    with pytest.raises(ValueError, match="changed after preparation"):
        session.begin(session.manifest_id, 3)
    assert access.owner is None


def binding_fixture(monkeypatch):
    """Keep allocation/lease/session code actual; replace native communicator setup."""
    session, access = session_fixture()
    geometry = preparation.ShardGeometry(2, 1, 1, ((0,), (1,)), 1, 3, 2)
    identity = {"role": "policy", "rank": 0, "identity": {"physical_node": "fixture"}}
    token = access.acquire("preparation")
    worker = SimpleNamespace(_policy_weight_access=access)
    worker._shard_preparation = preparation.LocalPreparation(
        "fixture",
        geometry,
        0,
        session.runner.sources,
        {},
        {},
        storage_versions(session.runner.sources),
        token,
        identity,
    )
    plan = SimpleNamespace(
        preparation_id="fixture",
        geometry=geometry,
        options=preparation.PreparationOptions(128, 17, 64, 0),
        identity_rows=(identity,),
        plan_id="bound-plan",
        schedule=None,
        expert_views=None,
        dense_plan=None,
        replica_plan=None,
        endpoints=None,
    )

    def native_factory(target, *args, **kwargs):
        target._shard_stream_session = session
        return {"manifest_id": session.manifest_id}

    monkeypatch.setattr(preparation, "prepare_native_shard_worker", native_factory)
    monkeypatch.setattr(preparation, "borrowed_policy_groups", lambda *args: ({}, {}, ()))
    return worker, plan, session


def test_actual_binding_transfers_ownership_and_unstarted_cleanup_releases(monkeypatch):
    worker, plan, session = binding_fixture(monkeypatch)
    receipts = []
    result = preparation.bind_live_preparation(worker, plan, None, receipts.append)
    assert result["source_lease_transferred"] is True
    assert result["allocation_after"]["cuda_measured"] is False
    assert worker._shard_preparation.token is None
    assert session.token is not None
    with pytest.raises(RuntimeError, match="already owned"):
        worker._policy_weight_access.acquire("ppo")
    with pytest.raises(ValueError, match="fresh"):
        preparation.bind_live_preparation(worker, plan, None, receipts.append)
    preparation.close_live_preparation(worker, "fixture")
    assert worker._policy_weight_access.owner is None
    assert not hasattr(worker, "_shard_stream_session")


@pytest.mark.parametrize("primary_failure", [False, True])
def test_allocation_and_sink_failures_preserve_primary_and_cleanup(monkeypatch, primary_failure):
    worker, plan, session = binding_fixture(monkeypatch)

    def failed_factory(*args, **kwargs):
        raise ValueError("native group failure")

    if primary_failure:
        monkeypatch.setattr(preparation, "prepare_native_shard_worker", failed_factory)
    calls = 0

    def allocation(device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("allocation readback failed")
        return {"cuda_measured": False}

    captured = []

    def failed_sink(receipt):
        captured.append(receipt.copy())
        raise OSError("sink failed")

    monkeypatch.setattr(preparation, "_allocation_state", allocation)
    with pytest.raises(
        (ValueError, RuntimeError), match="native group failure" if primary_failure else "allocation readback failed"
    ) as caught:
        preparation.bind_live_preparation(worker, plan, None, failed_sink)
    assert any("sink failed" in note for note in caught.value.__notes__)
    assert captured[0]["allocation_readback_error"] == "RuntimeError: allocation readback failed"
    assert worker._shard_preparation.phase is preparation.PreparationPhase.FAILED
    with pytest.raises(ValueError, match="fresh"):
        preparation.bind_live_preparation(worker, plan, None, captured.append)
    preparation.close_live_preparation(worker, "fixture")
    assert worker._policy_weight_access.owner is None


@pytest.mark.parametrize("role", ["policy", "receiver"])
def test_actual_collectors_use_named_unequal_ep_error(monkeypatch, role):
    worker, geometry, parallel = policy_worker(monkeypatch)
    parallel.get_expert_model_parallel_world_size = lambda: 4
    with pytest.raises(preparation.UnequalExpertParallelism, match="K10_EQUAL_EP_REQUIRED"):
        if role == "policy":
            preparation.collect_policy_preparation(worker, "actual", geometry, parallel, {"rank": 0}, lambda row: None)
        else:
            preparation.collect_receiver_preparation(worker, "actual", geometry, 0, 0, 4, {"rank": 0})
    assert worker._policy_weight_access.owner is None
    assert not hasattr(worker, "_shard_preparation")


@pytest.mark.parametrize("role", ["policy", "receiver"])
def test_collectors_keep_world_mismatch_distinct_from_unequal_ep(monkeypatch, role):
    worker, geometry, parallel = policy_worker(monkeypatch)
    monkeypatch.setattr(preparation.dist, "get_world_size", lambda: 3)
    with pytest.raises(ValueError, match="world") as caught:
        if role == "policy":
            preparation.collect_policy_preparation(worker, "actual", geometry, parallel, {"rank": 0}, lambda row: None)
        else:
            preparation.collect_receiver_preparation(worker, "actual", geometry, 0, 0, 2, {"rank": 0})
    assert type(caught.value) is ValueError


def test_source_failure_retains_allocation_receipt_before_lease_release(monkeypatch):
    worker, geometry, parallel = policy_worker(monkeypatch)

    def broken_bridge(modules):
        raise ValueError("source export failed")

    worker.bridge.get_conversion_tasks = broken_bridge
    captured = []
    with pytest.raises(ValueError, match="source export failed"):
        preparation.collect_policy_preparation(worker, "actual", geometry, parallel, {"rank": 0}, captured.append)
    assert captured[0]["phase"] == "source-metadata-failed"
    assert captured[0]["allocation_before"]["cuda_measured"] is False
    assert captured[0]["allocation_after"]["cuda_measured"] is False
    assert worker._policy_weight_access.owner is None


def test_unstarted_binding_cleanup_detects_mutation_and_still_releases(monkeypatch):
    worker, plan, session = binding_fixture(monkeypatch)
    preparation.bind_live_preparation(worker, plan, None, lambda receipt: None)
    next(iter(session.runner.sources.values())).add_(1)
    with pytest.raises(RuntimeError, match="Frozen learner source changed"):
        preparation.close_live_preparation(worker, "fixture")
    assert worker._policy_weight_access.owner is None
    assert not hasattr(worker, "_shard_preparation")
    assert not hasattr(worker, "_shard_stream_session")


@pytest.mark.parametrize("replace_storage", [False, True])
def test_live_inventory_accepts_updated_values_but_rejects_replaced_bridge_storage(monkeypatch, replace_storage):
    worker, geometry, parallel = policy_worker(monkeypatch)
    preparation.collect_policy_preparation(worker, "actual", geometry, parallel, {"rank": 0}, lambda row: None)
    state = worker._shard_preparation
    conversion = worker.bridge.get_conversion_tasks(worker.actor_module)[0]
    try:
        if replace_storage:
            conversion.param_weight = conversion.param_weight.clone()
            with pytest.raises(ValueError, match="Live learner inventory"):
                preparation.validate_live_inventory(worker, state)
        else:
            conversion.param_weight.add_(1)
            preparation.validate_live_inventory(worker, state)
    finally:
        preparation.close_live_preparation(worker, "actual")


@pytest.mark.parametrize("completed, publication", [(None, 1), (1, 2), (True, 1)])
def test_live_inventory_rejects_claimed_publication_without_actual_completed_update(
    monkeypatch, completed, publication
):
    worker, geometry, parallel = policy_worker(monkeypatch)
    preparation.collect_policy_preparation(worker, "actual", geometry, parallel, {"rank": 0}, lambda row: None)
    worker._completed_update = completed
    try:
        with pytest.raises(ValueError, match="actual completed learner update"):
            preparation.validate_live_inventory(worker, worker._shard_preparation, publication)
    finally:
        preparation.close_live_preparation(worker, "actual")
