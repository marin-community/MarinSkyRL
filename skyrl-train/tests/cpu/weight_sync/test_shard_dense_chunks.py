from dataclasses import replace

import pytest
import ray
import torch

from skyrl_train.weight_sync.shard_stream import dense_chunks
from tests.cpu.weight_sync.test_shard_stream import StreamActor, fixture_plan


class LargeDenseActor(StreamActor):
    def __init__(self, rank, payload, directory):
        super().__init__(rank, payload, directory)
        trainers = payload[0]
        if rank < len(trainers):
            layer = trainers[rank].pp
            self.sources[f"layer{layer}.router"] = torch.arange(139, dtype=torch.bfloat16) - 70
            self.sources[f"layer{layer}.bias"] = torch.arange(139, dtype=torch.float32) - 70
            self.sources[f"layer{layer}.bias"][0] = -0.0
            self.original = {name: tensor.view(torch.uint8).clone() for name, tensor in self.sources.items()}


@pytest.mark.parametrize("dtype,capacity", [("bfloat16", 127), ("float32", 127), ("bfloat16", 128), ("float32", 128)])
def test_dense_chunks_preserve_every_source_and_destination_element(dtype, capacity):
    item = fixture_plan(1, 1, 1)[4][0]
    item = replace(item, source=replace(item.source, numel=139, source_offset=5, hf_offset=11, wire_dtype=dtype))
    chunks = tuple(dense_chunks(item, capacity))
    width = 2 if dtype == "bfloat16" else 4
    assert len(chunks) > 1
    assert all(chunk.source.numel * width <= capacity for chunk in chunks)
    assert [
        offset
        for chunk in chunks
        for offset in range(chunk.source.source_offset, chunk.source.source_offset + chunk.source.numel)
    ] == list(range(5, 144))
    assert [
        offset
        for chunk in chunks
        for offset in range(chunk.source.hf_offset, chunk.source.hf_offset + chunk.source.numel)
    ] == list(range(11, 150))


def test_actual_large_dense_install_has_tails_and_preserves_router_widening(tmp_path):
    trainers, receivers, views, schedule, dense, shapes = fixture_plan(1, 1, 2)
    dense = tuple(
        replace(item, source=replace(item.source, numel=139))
        if item.source.hf_name.endswith((".mlp.router.weight", ".mlp.router.bias"))
        else item
        for item in dense
    )
    shapes = {
        name: ((139,), dtype) if name.endswith((".mlp.router.weight", ".mlp.router.bias")) else (shape, dtype)
        for name, (shape, dtype) in shapes.items()
    }
    payload = trainers, receivers, views, schedule, dense, shapes
    ray.init(num_cpus=4, include_dashboard=False)
    actors = []
    try:
        actor_type = ray.remote(num_cpus=1)(LargeDenseActor)
        actors = [actor_type.remote(rank, payload, str(tmp_path)) for rank in range(4)]
        ray.get([actor.initialize.remote() for actor in actors], timeout=60)
        results = ray.get([actor.run.remote() for actor in actors], timeout=60)
        assert all(row["sources_unchanged"] for row in results)
        for receiver in results[2:]:
            for layer in range(2):
                for suffix, expected in (
                    ("weight", (torch.arange(139, dtype=torch.bfloat16) - 70).float()),
                    ("bias", torch.arange(139, dtype=torch.float32) - 70),
                ):
                    if suffix == "bias":
                        expected[0] = -0.0
                    name = f"model.layers.{layer}.mlp.router.{suffix}"
                    assert receiver["parameter_bytes"][name] == expected.view(torch.uint8).numpy().tobytes().hex()
            chunks = [row for row in receiver["rows"] if row["phase"] == "dense"]
            assert any(row["wire_bytes"] == 22 and row["bytes"] == 44 for row in chunks)
            assert any(row["wire_bytes"] == 44 and row["bytes"] == 44 for row in chunks)
            assert all(row["wire_bytes"] <= 128 for row in chunks)
    finally:
        if actors:
            ray.get([actor.close.remote() for actor in actors], timeout=30)
        for actor in actors:
            ray.kill(actor, no_restart=True)
        ray.shutdown()
