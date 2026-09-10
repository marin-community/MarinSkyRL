"""Real Ray dispatch and custom Gloo groups, with complete tiny tensor bytes."""

from datetime import timedelta
from itertools import product
from pathlib import Path

import pytest
import ray
import torch
import torch.distributed as dist

from skyrl_train.distributed.utils import init_custom_process_group
from skyrl_train.weight_sync.frozen_source_views import FrozenSourceSlice
from skyrl_train.weight_sync.shard_group_schedule import (
    ExpertEntry,
    ReceiverRank,
    TrainerRank,
    build_shard_group_schedule,
)
from skyrl_train.weight_sync.shard_source_inventory import DenseSourceRank, LocalExpertSource, dense_stream_plan
from skyrl_train.weight_sync.shard_stream import ShardStreamRank


def fixture_plan(dp_count, ep_count, replicas):
    trainers = tuple(
        TrainerRank(i, dp, pp, ep) for i, (dp, pp, ep) in enumerate(product(range(dp_count), range(2), range(ep_count)))
    )
    receivers = tuple(
        ReceiverRank(i, replica, ep) for i, (replica, ep) in enumerate(product(range(replicas), range(ep_count)))
    )
    views = tuple(
        LocalExpertSource(
            ExpertEntry(
                f"layer{layer}.{projection}.expert{expert}",
                layer,
                layer,
                expert,
                projection,
                24 if projection == "fc1" else 12,
            ),
            f"layer{layer}.{projection}.expert{expert}",
            (4, 3) if projection == "fc1" else (3, 2),
        )
        for layer, expert, projection in product(range(2), range(ep_count), ("fc1", "fc2"))
    )
    schedule = build_shard_group_schedule(
        trainers,
        receivers,
        tuple(item.entry for item in views),
        trainer_ep=ep_count,
        receiver_ep=ep_count,
        layers_by_pp=((0,), (1,)),
        num_experts=ep_count,
    )
    dense_rows, shapes = [], {}
    for trainer in trainers:
        layer = trainer.pp
        qname, router, bias = (
            f"model.layers.{layer}.{part}"
            for part in ("self_attn.q_proj.weight", "mlp.router.weight", "mlp.router.bias")
        )
        rows = (
            FrozenSourceSlice(qname, 0, 2, "bfloat16", f"layer{layer}.qkv", 0, False),
            FrozenSourceSlice(qname, 2, 2, "bfloat16", f"layer{layer}.qkv", 4, False),
            FrozenSourceSlice(router, 0, 2, "bfloat16", f"layer{layer}.router", 0, False),
            FrozenSourceSlice(bias, 0, 2, "float32", f"layer{layer}.bias", 0, False),
        )
        dense_rows.append(DenseSourceRank(trainer, rows))
        shapes.update({qname: ((4,), "bfloat16"), router: ((2,), "bfloat16"), bias: ((2,), "float32")})
    dense = dense_stream_plan(dense_rows, receivers, shapes, expert_parallel_size=ep_count)
    return trainers, receivers, views, schedule, dense, shapes


class StreamActor:
    def __init__(self, rank, payload, directory, alias_scratch=False):
        self.rank = rank
        self.payload = payload
        self.directory = Path(directory)
        trainers, receivers, views, schedule, dense, shapes = payload
        self.sources, self.parameters, self.maps = {}, {}, {}
        if rank < len(trainers):
            trainer = trainers[rank]
            # Deliberately divergent replicas expose accidental writes into a
            # non-root learner; this fixture does not grant replica identity.
            shift = trainer.dp * 64 + trainer.pp * 16 + trainer.ep * 4
            for view in views:
                if view.entry.pp == trainer.pp and view.entry.expert == trainer.ep:
                    self.sources[view.source_key] = (
                        torch.arange(view.entry.nbytes // 2, dtype=torch.bfloat16).reshape(view.shape) + shift
                    )
            self.sources[f"layer{trainer.pp}.qkv"] = (
                torch.arange(6, dtype=torch.bfloat16) + trainer.pp * 16 + trainer.dp * 64
            )
            self.sources[f"layer{trainer.pp}.router"] = torch.tensor([1.5, -2.0], dtype=torch.bfloat16)
            self.sources[f"layer{trainer.pp}.bias"] = torch.tensor([-0.0, 1.0], dtype=torch.float32)
        else:
            receiver = receivers[rank - len(trainers)]
            for layer in range(2):
                prefix = f"model.layers.{layer}.mlp.experts.routed_experts"
                self.parameters[prefix + ".w13_weight"] = torch.full((1, 4, 3), -1, dtype=torch.bfloat16)
                self.parameters[prefix + ".w2_weight"] = torch.full((1, 3, 2), -1, dtype=torch.bfloat16)
                self.maps[prefix] = [0 if ep == receiver.ep else -1 for ep in range(len(schedule.groups))]
            for name, (shape, dtype) in shapes.items():
                self.parameters[name] = torch.zeros(
                    shape, dtype=torch.float32 if name.endswith(".mlp.router.weight") else getattr(torch, dtype)
                )
        self.original = {name: tensor.view(torch.uint8).clone() for name, tensor in self.sources.items()}
        self.scratch = torch.empty(128, dtype=torch.uint8)
        if alias_scratch and self.sources:
            self.scratch = next(iter(self.sources.values())).view(torch.uint8).flatten()
        self.groups = {}
        self.local_group = None
        dist.init_process_group(
            "gloo",
            init_method=f"file://{directory}/default-{rank}",
            rank=0,
            world_size=1,
            timeout=timedelta(seconds=20),
        )

    def initialize(self):
        trainers, receivers, views, schedule, dense, shapes = self.payload
        for group in schedule.groups:
            if self.rank in group.members:
                self.groups[group.ep] = init_custom_process_group(
                    "gloo",
                    init_method=f"file://{self.directory}/expert-{group.ep}",
                    rank=group.members.index(self.rank),
                    world_size=len(group.members),
                    group_name=f"expert-{group.ep}",
                    timeout=timedelta(seconds=20),
                )
        if self.rank >= len(trainers):
            receiver = receivers[self.rank - len(trainers)]
            local = tuple(len(trainers) + row.rank for row in receivers if row.replica == receiver.replica)
            self.local_group = init_custom_process_group(
                "gloo",
                init_method=f"file://{self.directory}/local-{receiver.replica}",
                rank=local.index(self.rank),
                world_size=len(local),
                group_name=f"local-{receiver.replica}",
                timeout=timedelta(seconds=20),
            )
        self.runner = ShardStreamRank(
            self.rank,
            schedule,
            views,
            dense,
            self.sources,
            self.parameters,
            self.maps,
            self.scratch,
            self.groups,
            self.local_group,
        )
        self.runner.begin(manifest_id="tiny", publication_id=3)
        return self.rank

    def run(self):
        result = self.runner.run(manifest_id="tiny", publication_id=3)
        assert all(torch.equal(value.view(torch.uint8), self.original[name]) for name, value in self.sources.items())
        result["sources_unchanged"] = True
        result["parameters"] = {name: value.tolist() for name, value in self.parameters.items()}
        result["parameter_bytes"] = {
            name: value.view(torch.uint8).numpy().tobytes().hex() for name, value in self.parameters.items()
        }
        result["bias_bits"] = {
            name: value.view(torch.int32).tolist()
            for name, value in self.parameters.items()
            if name.endswith(".mlp.router.bias")
        }
        return result

    def repeat(self):
        return self.runner.run(manifest_id="tiny", publication_id=3)

    def close(self):
        if self.local_group is not None:
            dist.destroy_process_group(self.local_group)
        for group in self.groups.values():
            dist.destroy_process_group(group)
        dist.destroy_process_group()
        return self.rank


@pytest.mark.parametrize(("dp_count", "ep_count", "replicas"), [(2, 1, 2), (1, 2, 2)])
def test_actual_ray_dispatch_preserves_nonroot_sources_and_installs_both_streams(
    tmp_path, dp_count, ep_count, replicas
):
    payload = fixture_plan(dp_count, ep_count, replicas)
    trainers, receivers, views, schedule, dense, shapes = payload
    count = len(trainers) + len(receivers)
    ray.init(num_cpus=count, include_dashboard=False)
    actors = []
    try:
        actor_type = ray.remote(num_cpus=1)(StreamActor)
        actors = [actor_type.remote(rank, payload, str(tmp_path)) for rank in range(count)]
        assert sorted(ray.get([actor.initialize.remote() for actor in actors], timeout=90)) == list(range(count))
        results = ray.get([actor.run.remote() for actor in actors], timeout=60)
        assert all(row["sources_unchanged"] for row in results)
        for receiver in receivers:
            result = results[len(trainers) + receiver.rank]
            for layer in range(2):
                shift = (layer % dp_count) * 64 + layer * 16 + receiver.ep * 4
                prefix = f"model.layers.{layer}.mlp.experts.routed_experts"
                assert result["parameters"][prefix + ".w13_weight"] == [
                    (torch.arange(12).reshape(4, 3) + shift).tolist()
                ]
                assert result["parameters"][prefix + ".w2_weight"] == [(torch.arange(6).reshape(3, 2) + shift).tolist()]
                for suffix, shape in ((".w13_weight", (1, 4, 3)), (".w2_weight", (1, 3, 2))):
                    expected = (
                        torch.arange(12 if suffix == ".w13_weight" else 6, dtype=torch.bfloat16).reshape(shape) + shift
                    )
                    assert (
                        result["parameter_bytes"][prefix + suffix] == expected.view(torch.uint8).numpy().tobytes().hex()
                    )
                qshift = layer * 16 + (layer % dp_count) * 64
                assert result["parameters"][f"model.layers.{layer}.self_attn.q_proj.weight"] == [
                    qshift + i for i in (0, 1, 4, 5)
                ]
                assert result["parameters"][f"model.layers.{layer}.mlp.router.weight"] == [1.5, -2.0]
                assert result["bias_bits"][f"model.layers.{layer}.mlp.router.bias"] == [-(2**31), 1065353216]
                for name, expected in (
                    (
                        f"model.layers.{layer}.self_attn.q_proj.weight",
                        torch.tensor([qshift + i for i in (0, 1, 4, 5)], dtype=torch.bfloat16),
                    ),
                    (f"model.layers.{layer}.mlp.router.weight", torch.tensor([1.5, -2.0], dtype=torch.float32)),
                    (f"model.layers.{layer}.mlp.router.bias", torch.tensor([-0.0, 1.0], dtype=torch.float32)),
                ):
                    assert result["parameter_bytes"][name] == expected.view(torch.uint8).numpy().tobytes().hex()

        with pytest.raises(ray.exceptions.RayTaskError, match="fresh prepared publication"):
            ray.get(actors[0].repeat.remote(), timeout=10)
    finally:
        if actors:
            ray.get([actor.close.remote() for actor in actors], timeout=30)
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()


def test_workspace_alias_is_rejected_before_communicator_or_parameter_write():
    trainers, receivers, views, schedule, dense, shapes = fixture_plan(2, 1, 2)
    first = views[0]
    tensor = torch.arange(12, dtype=torch.bfloat16).reshape(first.shape)
    original = tensor.view(torch.uint8).clone()
    with pytest.raises(ValueError, match="overlap"):
        ShardStreamRank(
            0, schedule, views, dense, {first.source_key: tensor}, {}, {}, tensor.view(torch.uint8).flatten(), {}, None
        )
    assert torch.equal(tensor.view(torch.uint8), original)
