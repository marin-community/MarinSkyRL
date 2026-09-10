"""Full installed-byte replay over real Ray/custom Gloo with corruption controls."""

import asyncio

import pytest
import ray
import torch

from skyrl_train.weight_sync.shard_replica_proof import build_replica_plan
from tests.cpu.weight_sync.test_shard_native_factory import FactoryActor
from tests.cpu.weight_sync.test_shard_stream import fixture_plan


class ReplayActor(FactoryActor):
    def prepare_epoch(self, plan, namespace):
        self.directory = self.directory.parent / namespace
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.prepare(plan)

    def corrupt(self, kind):
        suffix = {
            "expert": ".w13_weight",
            "dense": ".q_proj.weight",
            "router": ".router.weight",
            "bias": ".router.bias",
        }[kind]
        name = next(name for name in self.parameters if name.endswith(suffix))
        # Simulate a storage fault, which does not increment Torch's mutation
        # version. The full-byte callback must detect it independently.
        target = self.parameters[name].data.view(torch.uint8).reshape(-1)
        offset = 3 if kind == "bias" else 0
        target[offset] ^= 128 if kind == "bias" else 1
        return name

    def extra_parameter(self):
        self.parameters["unplanned.parameter"] = torch.zeros(3, dtype=torch.bfloat16)


@pytest.fixture(scope="module")
def replay_ranks(tmp_path_factory):
    directory = tmp_path_factory.mktemp("complete-shard-replay")
    payload = fixture_plan(2, 1, 2)
    ray.init(num_cpus=6, include_dashboard=False)
    cls = ray.remote(num_cpus=1)(ReplayActor)
    actors = [cls.remote(rank, payload, str(directory)) for rank in range(6)]
    catalogue = ray.get([actor.catalogue.remote() for actor in actors[:4]], timeout=60)
    plan = build_replica_plan(payload[0], payload[3], tuple(catalogue))
    yield actors, plan
    ray.get([actor.shutdown.remote() for actor in actors], timeout=30)
    for actor in actors:
        ray.kill(actor, no_restart=True)
    ray.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [None, "expert", "dense", "router", "bias"])
async def test_complete_replay_detects_fault_without_rewriting_installed_bytes(replay_ranks, fault):
    actors, plan = replay_ranks
    prepared = await asyncio.gather(*(actor.prepare_epoch.remote(plan, f"replay-{fault}") for actor in actors))
    manifest = prepared[0]["manifest_id"]
    assert {row["manifest_id"] for row in prepared} == {manifest}
    try:
        await asyncio.gather(*(actor.phase.remote("begin", manifest) for actor in actors))
        await asyncio.gather(*(actor.phase.remote("verify_replicas", manifest) for actor in actors[:4]))
        await asyncio.gather(*(actor.phase.remote("run", manifest) for actor in actors))
        ready = await asyncio.gather(*(actor.phase.remote("prepare_replay", manifest) for actor in actors))
        assert [row["expected_bytes"] for row in ready] == [0, 0, 0, 0, 120, 120]
        if fault is not None:
            await actors[4].corrupt.remote(fault)
        before = await asyncio.gather(*(actor.inspect.remote() for actor in actors))
        results = await asyncio.gather(*(actor.phase.remote("replay", manifest) for actor in actors))
        after = await asyncio.gather(*(actor.inspect.remote() for actor in actors))
        assert before == after
        assert all(row["unchanged"] for row in after)
        assert all(
            row["compared_bytes"] == 120 and row["expected_bytes"] == 120 and row["coverage"] == 1.0
            for row in results[4:]
        )
        assert all(row["memory_after"]["cuda_measured"] is False for row in results)
        assert all(row["inter_group_collective_payload_bytes"] == 112 for row in results)
        assert [row["local_fanout_collective_payload_bytes"] for row in results] == [0, 0, 0, 0, 40, 40]
        assert all(row["physical_nic_bytes"] is None for row in results)
        assert results[5]["mismatches"] == 0
        assert results[4]["mismatches"] == (0 if fault is None else 1)
        assert results[4]["phase"] == ("verified" if fault is None else "failed")
        with pytest.raises(ray.exceptions.RayTaskError, match="complete prechecked"):
            await actors[4].phase.remote("replay", manifest)
    finally:
        await asyncio.gather(*(actor.phase.remote("close", manifest) for actor in actors))


@pytest.mark.asyncio
async def test_independent_inventory_rejects_unrepresented_installed_parameter(replay_ranks):
    actors, plan = replay_ranks
    await actors[4].extra_parameter.remote()
    prepared = await asyncio.gather(*(actor.prepare_epoch.remote(plan, "replay-unplanned") for actor in actors))
    manifest = prepared[0]["manifest_id"]
    assert prepared[4]["expected_receiver_bytes"] == 126
    try:
        await asyncio.gather(*(actor.phase.remote("begin", manifest) for actor in actors))
        await asyncio.gather(*(actor.phase.remote("verify_replicas", manifest) for actor in actors[:4]))
        await asyncio.gather(*(actor.phase.remote("run", manifest) for actor in actors))
        with pytest.raises(ray.exceptions.RayTaskError, match="missed installed receiver bytes"):
            await actors[4].phase.remote("prepare_replay", manifest)
    finally:
        await asyncio.gather(*(actor.phase.remote("close", manifest) for actor in actors))
