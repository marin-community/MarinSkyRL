from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def benchmark_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "grug_fixed_replay_benchmark.py"
    spec = importlib.util.spec_from_file_location("grug_fixed_replay_benchmark", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay_shards(module):
    rows = torch.arange(module.HEADLINE_SEQUENCES, dtype=torch.int64).reshape(-1, 1)
    return [
        {name: rows[start : start + 128].clone() for name in module.TENSOR_FIELDS}
        for start in range(0, module.HEADLINE_SEQUENCES, 128)
    ]


@pytest.mark.parametrize(
    ("expert_parallel_size", "data_parallel_size", "rows_per_rank"),
    ((1, 32, 128), (8, 4, 1024)),
)
def test_headline_batches_follow_mesh_dispatch_replication(
    benchmark_module, expert_parallel_size, data_parallel_size, rows_per_rank
):
    module = benchmark_module
    manifest = {"batch_metadata": {"global_step": 1}}
    batches = module.make_rank_batches(
        manifest,
        replay_shards(module),
        "headline",
        expert_parallel_size,
    )

    assert len(batches) == 32
    assert module.expected_microbatches("headline", expert_parallel_size) == rows_per_rank
    for data_rank in range(data_parallel_size):
        replicas = batches[data_rank * expert_parallel_size : (data_rank + 1) * expert_parallel_size]
        assert len(replicas) == expert_parallel_size
        assert {batch.batch_size for batch in replicas} == {rows_per_rank}
        assert len({module.field_identity(batch)[0]["sequences"] for batch in replicas}) == 1
        assert {batch.metadata["grug_benchmark_data_rank"] for batch in replicas} == {data_rank}
        assert [batch.metadata["grug_benchmark_ep_replica"] for batch in replicas] == list(range(expert_parallel_size))

    logical_rows = torch.cat(
        [batches[data_rank * expert_parallel_size]["sequences"] for data_rank in range(data_parallel_size)]
    )
    assert torch.equal(logical_rows.reshape(-1), torch.arange(module.HEADLINE_SEQUENCES))


def test_headline_topology_requires_four_complete_nodes(benchmark_module):
    topology = [{"rank": rank, "phys_uuid": f"gpu-{rank}", "host": f"node-{rank // 8}"} for rank in range(32)]

    benchmark_module.assert_topology(topology, "headline")

    topology[-1]["host"] = "node-extra"
    with pytest.raises(RuntimeError, match="four complete 8-GPU hosts|expected 4 complete 8-GPU hosts"):
        benchmark_module.assert_topology(topology, "headline")


def test_representative_gradient_aggregation_removes_only_ep_replication(benchmark_module):
    timed = []
    for rank in range(8):
        timed.append(
            {
                "representative_gradients": {
                    "model.layers.0.input_layernorm.weight": {
                        "local_numel": 10,
                        "l2_norm": 2.0,
                        "max_abs": 0.5,
                    },
                    "model.layers.0.mlp.experts.down_proj.weight": {
                        "local_numel": 10 + rank,
                        "l2_norm": float(rank + 1),
                        "max_abs": float(rank + 1) / 10,
                    },
                }
            }
        )

    aggregated = benchmark_module.aggregate_representative_gradients(timed, expert_parallel_size=8)

    assert aggregated["model.layers.0.input_layernorm.weight"] == {
        "numel": 10,
        "l2_norm": 2.0,
        "max_abs": 0.5,
        "ep_replication_divisor": 8,
    }
    expert = aggregated["model.layers.0.mlp.experts.down_proj.weight"]
    assert expert["numel"] == sum(range(10, 18))
    assert expert["l2_norm"] == pytest.approx(sum(value**2 for value in range(1, 9)) ** 0.5)
    assert expert["max_abs"] == 0.8
    assert expert["ep_replication_divisor"] == 1
