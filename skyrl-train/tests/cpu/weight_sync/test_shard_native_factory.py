import ray
import torch

from skyrl_train.weight_sync.policy_weight_access import PolicyWeightAccess
from skyrl_train.weight_sync.shard_group_factory import GroupEndpoint
from skyrl_train.weight_sync.shard_native_factory import prepare_native_shard_worker, required_group_memberships
from skyrl_train.weight_sync.shard_replica_proof import ReplicaCatalogue, ReplicaTensor, build_replica_plan
from skyrl_train.weight_sync.shard_session import worker_shard_call
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
