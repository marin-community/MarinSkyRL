"""Actual external-DP actor construction and complete readback regrouping."""

from types import SimpleNamespace

import pytest

from skyrl_train.weight_sync.initial_readback import validate_initial_readback
from tests.cpu.weight_sync.test_readback_diagnostics import _readbacks
from skyrl_train.weight_sync.receiver_readback_rpc import group_external_dp_workers, read_all_receiver_workers
from tests.cpu.test_engine_placement_strategy import inference_scheduler as inference_scheduler


@pytest.mark.asyncio
async def test_actual_factory_external_dp_actors_retain_all_eight_worker_origins(request):
    scheduler = request.getfixturevalue("inference_scheduler")
    wrappers = scheduler.launch(num_inference_engines=1, data_parallel_size=8, expert_parallel_size=8, node_local=True)
    assert len(wrappers) == len(scheduler.actors) == 8
    actor_rows = []
    for actor in scheduler.actors:
        rank = actor.kwargs["data_parallel_rank"]
        size = actor.kwargs["data_parallel_size"]
        identity = rank.to_bytes(2, "little")

        class CoreBoundary:
            core_engines = [identity]
            engine_ranks_managed = [rank]

            async def _call_utility_async(self, utility, method, timeout, args, kwargs, *, engine):
                assert engine == identity and utility == "collective_rpc"
                return [{"rank": rank, "world_size": size, "native_payload": f"worker-{rank}"}]

        engine = SimpleNamespace(
            engine_core=CoreBoundary(),
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(
                    data_parallel_size=size,
                    data_parallel_size_local=1,
                    data_parallel_index=rank,
                    data_parallel_rank_local=None,
                    local_engines_only=True,
                )
            ),
        )
        actor_rows.append(await read_all_receiver_workers(engine, "read_weight_sync_environment"))
    geometry = {"receiver_engines": 1, "receiver_ranks_per_engine": 8, "receiver_parallel": {"data_parallel_size": 8}}
    rows = group_external_dp_workers(actor_rows, geometry)
    assert len(rows) == 1 and [row["rank"] for row in rows[0]] == list(range(8))
    assert [row["native_payload"] for row in rows[0]] == [f"worker-{rank}" for rank in range(8)]
    assert [row["receiver_transport"]["managed_dp_ranks"] for row in rows[0]] == [[rank] for rank in range(8)]
    assert all(row["world_size"] == 8 for row in rows[0])


@pytest.mark.parametrize("damage", ["missing_actor", "duplicate_rank", "wrong_actor_order"])
def test_missing_or_wrong_external_actor_never_satisfies_complete_coverage(damage):
    rows = [[{"rank": rank}] for rank in range(8)]
    geometry = {"receiver_engines": 1, "receiver_ranks_per_engine": 8, "receiver_parallel": {"data_parallel_size": 8}}
    if damage == "missing_actor":
        rows.pop()
    elif damage == "duplicate_rank":
        rows[4] = [{"rank": 3}]
    else:
        rows[0], rows[1] = rows[1], rows[0]
    with pytest.raises(ValueError, match="external DP actors|factory slice"):
        group_external_dp_workers(rows, geometry)


def test_outer_grouping_does_not_hide_wrong_native_world_size():
    policy, _ = _readbacks()
    geometry = {
        "policy_ranks": 1,
        "tp_rank": 1,
        "pp_rank": 1,
        "ep_rank": 1,
        "receiver_engines": 1,
        "receiver_ranks_per_engine": 8,
        "receiver_parallel": {"data_parallel_size": 8},
    }
    rows = [
        [
            {
                "rank": rank,
                "world_size": 8,
                "parallel_config": {"data_parallel_size": 8},
                "environment": policy[0]["environment"],
            }
        ]
        for rank in range(8)
    ]
    grouped = group_external_dp_workers(rows, geometry)
    validate_initial_readback(policy, grouped, geometry)
    grouped[0][6]["world_size"] = 7
    with pytest.raises(ValueError, match="parallel geometry"):
        validate_initial_readback(policy, grouped, geometry)
