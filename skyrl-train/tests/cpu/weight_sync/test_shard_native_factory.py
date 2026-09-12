import asyncio
from functools import partial
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import ray
import torch

from skyrl_train.weight_sync.readback_diagnostics import persist_readback
from skyrl_train.weight_sync.policy_weight_access import PolicyWeightAccess
from skyrl_train.weight_sync.shard_group_factory import GroupEndpoint
from skyrl_train.weight_sync.shard_native_factory import prepare_native_shard_worker, required_group_memberships
from skyrl_train.weight_sync.shard_replica_proof import ReplicaCatalogue, ReplicaTensor, build_replica_plan
from skyrl_train.weight_sync.shard_session import worker_shard_call
from skyrl_train.weight_sync.shard_interval import GenerationBoundary, ShardLifecycle, run_shard_interval
from tests.cpu.weight_sync.test_shard_stream import StreamActor, fixture_plan


class FactoryActor(StreamActor):
    def __init__(self, rank, payload, directory):
        super().__init__(rank, payload, directory)
        self.access = PolicyWeightAccess() if self.sources else None
        if self.sources:
            native = payload[0][rank]
            for name, value in self.sources.items():
                if ".expert" in name or name.endswith(".qkv"):
                    value.sub_(native.dp * 64)
            self.original = {name: value.view(torch.uint8).clone() for name, value in self.sources.items()}

    def catalogue(self):
        return ReplicaCatalogue(
            self.rank,
            tuple(
                ReplicaTensor(name, tuple(value.shape), str(value.dtype).removeprefix("torch."), ".expert" in name)
                for name, value in sorted(self.sources.items())
            ),
        )

    def prepare(self, plan):
        _, _, views, schedule, dense, _ = self.payload
        groups = required_group_memberships(schedule, dense, plan)
        endpoints = tuple(
            GroupEndpoint(name, members, "gloo", f"file://{self.directory}/{name}", 20)
            for name, members in groups.items()
        )
        return prepare_native_shard_worker(
            self,
            self.rank,
            schedule,
            views,
            dense,
            plan,
            endpoints,
            sources=self.sources,
            parameters=self.parameters,
            expert_maps=self.maps,
            transfer_workspace=self.scratch,
            comparison_workspace=torch.empty(17, dtype=torch.bool) if self.sources else None,
            dense_chunk_bytes=64,
            policy_access=self.access,
            proof_capture=partial(persist_readback, str(self.directory), "source-proof"),
        )

    def phase(self, name, manifest):
        return worker_shard_call(self, name, manifest, 4)

    def inspect(self):
        return {
            "parameters": {
                name: tensor.view(torch.uint8).numpy().tobytes().hex() for name, tensor in self.parameters.items()
            },
            "unchanged": all(
                torch.equal(value.view(torch.uint8), self.original[name]) for name, value in self.sources.items()
            ),
        }

    def shutdown(self):
        import torch.distributed as dist

        dist.destroy_process_group()


def test_native_factory_binds_actual_groups_and_complete_replica_comparator(tmp_path):
    payload = fixture_plan(2, 1, 2)
    trainers, receivers, views, schedule, dense, shapes = payload
    ray.init(num_cpus=6, include_dashboard=False)
    actors = []
    try:
        cls = ray.remote(num_cpus=1)(FactoryActor)
        actors = [cls.remote(rank, payload, str(tmp_path)) for rank in range(6)]
        catalogue = ray.get([actor.catalogue.remote() for actor in actors[:4]], timeout=60)
        plan = build_replica_plan(trainers, schedule, tuple(catalogue))
        before = ray.get([actor.inspect.remote() for actor in actors], timeout=30)
        prepared = ray.get([actor.prepare.remote(plan) for actor in actors], timeout=60)
        assert ray.get([actor.inspect.remote() for actor in actors], timeout=30) == before
        for row in prepared:
            readiness = row["group_readiness"]
            assert readiness["phase"] == "groups-ready" and readiness["cuda_measured"] is False
            assert [group["name"] for group in readiness["groups"]] == list(row["group_memberships"])
            assert all(group["payload_bytes"] == 4 and group["seconds"] >= 0 for group in readiness["groups"])
            assert readiness["new_explicit_tensor_storage_bytes"] == 0
        assert len({row["manifest_id"] for row in prepared}) == 1
        manifest = prepared[0]["manifest_id"]
        assert [row["expected_receiver_bytes"] for row in prepared[4:]] == [120, 120]
        ray.get([actor.phase.remote("begin", manifest) for actor in actors], timeout=30)
        proofs = ray.get([actor.phase.remote("verify_replicas", manifest) for actor in actors[:4]], timeout=60)
        assert all(row["proof"]["mismatches"] == 0 for row in proofs)
        for row in proofs:
            receipt = row["replica_groups"]
            assert receipt["memory_after"]["cuda_measured"] is False
            assert receipt["proof_peak_extra_bytes"] is None
            assert receipt["retained_comparison_bytes"] == 17
            binding = receipt["durable_receipt"]
            raw = Path(binding["uri"]).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == binding["sha256"]
            persisted = json.loads(raw)
            assert persisted["unique_source_bytes"] == row["proof"]["expected_bytes"]
            assert persisted["phase"] == "source-proof-compared"

        ray.get([actor.phase.remote("run", manifest) for actor in actors], timeout=60)
        inspected = ray.get([actor.inspect.remote() for actor in actors], timeout=30)
        assert all(row["unchanged"] for row in inspected)
        assert inspected[4]["parameters"] == inspected[5]["parameters"]
        for layer in range(2):
            name = f"model.layers.{layer}.mlp.experts.routed_experts.w13_weight"
            expected = torch.arange(12, dtype=torch.bfloat16).reshape(1, 4, 3) + layer * 16
            assert inspected[4]["parameters"][name] == expected.view(torch.uint8).numpy().tobytes().hex()
        ray.get([actor.phase.remote("close", manifest) for actor in actors], timeout=30)
    finally:
        if actors:
            ray.get([actor.shutdown.remote() for actor in actors], timeout=30)
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


class PersistentFactoryActor(FactoryActor):
    def versioned_phase(self, name, manifest, version):
        return worker_shard_call(self, name, manifest, version)

    def update_sources(self):
        if self.access is None:
            return
        with self.access.hold("policy-train"):
            for value in self.sources.values():
                value.add_(1)


def test_persistent_groups_publish_updated_source_bytes_after_verified_finish(tmp_path):
    payload = fixture_plan(1, 2, 2)
    trainers, receivers, _, schedule, _, _ = payload
    count = len(trainers) + len(receivers)
    ray.init(num_cpus=count, include_dashboard=False)
    actors = []
    try:
        actor_type = ray.remote(num_cpus=1)(PersistentFactoryActor)
        actors = [actor_type.remote(rank, payload, str(tmp_path)) for rank in range(count)]
        catalogue = ray.get([actor.catalogue.remote() for actor in actors[: len(trainers)]], timeout=60)
        plan = build_replica_plan(trainers, schedule, tuple(catalogue))
        prepared = ray.get([actor.prepare.remote(plan) for actor in actors], timeout=60)
        manifest = prepared[0]["manifest_id"]
        snapshots = []
        for version in (4, 5):
            ray.get([actor.versioned_phase.remote("begin", manifest, version) for actor in actors], timeout=30)
            ray.get(
                [
                    actor.versioned_phase.remote("verify_replicas", manifest, version)
                    for actor in actors[: len(trainers)]
                ],
                timeout=60,
            )
            ray.get([actor.versioned_phase.remote("run", manifest, version) for actor in actors], timeout=60)
            for actor in actors:
                with pytest.raises(ray.exceptions.RayTaskError, match="fully replayed weights"):
                    ray.get(actor.versioned_phase.remote("finish", manifest, version), timeout=30)
            ray.get([actor.versioned_phase.remote("prepare_replay", manifest, version) for actor in actors], timeout=30)
            proof = ray.get([actor.versioned_phase.remote("replay", manifest, version) for actor in actors], timeout=60)
            assert all(row["mismatches"] == 0 and row["coverage"] == 1.0 for row in proof[len(trainers) :])
            snapshots.append(ray.get([actor.inspect.remote() for actor in actors[len(trainers) :]], timeout=30))
            finished = ray.get(
                [actor.versioned_phase.remote("finish", manifest, version) for actor in actors], timeout=30
            )
            assert all(row["groups_retained"] and row["phase"] == "prepared" for row in finished)
            if version == 4:
                ray.get([actor.update_sources.remote() for actor in actors], timeout=30)
        assert all(before["parameters"] != after["parameters"] for before, after in zip(*snapshots, strict=True))
        for actor in actors:
            try:
                ray.get(actor.versioned_phase.remote("begin", manifest, 5), timeout=30)
            except ray.exceptions.RayTaskError as error:
                assert "advance beyond the completed version" in str(error)
            else:
                raise AssertionError("Completed publication version was reused")
        asyncio.run(exercise_persistent_coordinator(actors, len(trainers), manifest, prepared))
    finally:
        if actors:
            ray.get([actor.shutdown.remote() for actor in actors], timeout=30)
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


async def exercise_persistent_coordinator(actors, trainer_count, manifest, prepared):
    class Policy:
        def async_run_ray_method(self, dispatch, method, manifest_id, publication_id):
            assert dispatch == "pass_through"
            phase = {"verify": "verify_replicas"}.get(method.split("_")[0], method.split("_")[0])
            return [
                actor.versioned_phase.remote(phase, manifest_id, publication_id) for actor in actors[:trainer_count]
            ]

    class Client:
        def __init__(self):
            self.generation_paused_event = asyncio.Event()

        @property
        def paused(self):
            return self.generation_paused_event.is_set()

        async def pause_generation(self, **kwargs):
            self.generation_paused_event.set()

        async def resume_generation(self, **kwargs):
            self.generation_paused_event.clear()

        async def phase(self, phase, manifest_id, publication_id):
            return await asyncio.gather(
                *[actor.versioned_phase.remote(phase, manifest_id, publication_id) for actor in actors[trainer_count:]]
            )

        async def begin_shard_stream(self, *args):
            return await self.phase("begin", *args)

        async def run_shard_stream(self, *args):
            return await self.phase("run", *args)

        async def finish_shard_stream(self, *args):
            return await self.phase("finish", *args)

        async def close_shard_stream(self, *args):
            return await self.phase("close", *args)

    async def replay(manifest_id, version):
        if version == 8:
            raise ValueError("Injected replay boundary failure")
        await asyncio.gather(
            *[actor.versioned_phase.remote("prepare_replay", manifest_id, version) for actor in actors]
        )
        rows = await asyncio.gather(*[actor.versioned_phase.remote("replay", manifest_id, version) for actor in actors])
        return rows[trainer_count:]

    client = Client()
    driver = SimpleNamespace(policy_model=Policy(), inference_engine_client=client)
    arguments = dict(
        replay=replay,
        policy_ranks=tuple(range(trainer_count)),
        receiver_ranks=tuple(range(trainer_count, len(actors))),
        expected_receiver_bytes={row["rank"]: row["expected_receiver_bytes"] for row in prepared[trainer_count:]},
        lifecycle=ShardLifecycle.RETAIN,
    )
    result = await run_shard_interval(driver, manifest, 6, **arguments)
    assert not client.paused and "policy_close" not in result
    assert all(row["groups_retained"] for row in result["policy_finish"] + result["receiver_finish"])
    assert result["phase_seconds"]["install"] >= 0 and result["phase_seconds"]["full_byte_replay"] >= 0
    await client.pause_generation()
    await run_shard_interval(driver, manifest, 7, generation_boundary=GenerationBoundary.DRIVER, **arguments)
    assert client.paused
    await client.resume_generation()
    with pytest.raises(ValueError, match="Injected replay boundary failure"):
        await run_shard_interval(driver, manifest, 8, **arguments)
    assert client.paused
    # Cleanup has released every learner lease even after a failed publication.
    await asyncio.gather(*[actor.update_sources.remote() for actor in actors[:trainer_count]])
