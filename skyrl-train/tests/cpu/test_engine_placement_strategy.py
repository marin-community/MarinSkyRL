"""Unit tests for inference-engine placement and startup configuration.

Multi-GPU TP/PP and dense DP replicas stay node-local. Verified TP=PP=1 EP replicas
use PACK, which can place a replica on one node or span nodes. Single-GPU engines
share a flat PACK group; hybrid and multi-GPU mp executors keep their existing layout.

uv run --frozen pytest skyrl-train/tests/cpu/test_engine_placement_strategy.py
"""

from dataclasses import asdict, replace
import sys
from types import SimpleNamespace

import msgpack
import pytest

from marinskyrl.inference_placement import InferenceWorkerPlacement
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines.placement import inference_bundle_nodes, verified_inference_replica_placements
from skyrl_train.inference_engines import ray_wrapped_inference_engine as factory
from skyrl_train.inference_engines.utils import ReservedRendezvousPorts
from skyrl_train.inference_engines.ray_wrapped_inference_engine import resolve_engine_max_model_len
from skyrl_train.utils.placement_geometry import colocated_engine_bundle_indices
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.utils.utils import (
    use_per_engine_strict_pack_pg,
)
from tests.cpu.util import example_dummy_config


@pytest.mark.parametrize(
    "engine_kwargs,rope_scaling,expected",
    [
        ({"max_model_len": 16384}, {"factor": 4, "original_max_position_embeddings": 8192}, 16384),
        ({}, {"factor": 4, "original_max_position_embeddings": 8192}, 32768),
        ({}, None, None),
    ],
)
def test_resolve_engine_max_model_len(engine_kwargs, rope_scaling, expected):
    assert resolve_engine_max_model_len(engine_kwargs, rope_scaling) == expected


@pytest.mark.parametrize(
    "tp,pp,dp,expected",
    [
        # Regression: multi-node TP=1 must not use per-engine STRICT_PACK, which scatters
        # 1-GPU bundles and starves the policy PACK placement group of whole nodes.
        (1, 1, 1, False),
        (2, 1, 1, True),  # de-risk geometry on ray/uni -> on-node STRICT_PACK
        # Regression (#232): TP=4 on 4-GPU nodes must still STRICT_PACK; a `gpus > gpus_per_node` gate
        # re-breaks cross-node TP all-reduce with a decode deadlock.
        (4, 1, 1, True),
        (1, 2, 1, True),  # PP=2 single TP -> multi-GPU engine, still needs on-node
        (2, 2, 1, True),  # TP*PP=4
        (1, 1, 4, True),  # DP4xEP4 on 4-GPU nodes -> on-node STRICT_PACK
        (1, 1, 8, True),  # DP8xEP8 on 8-GPU nodes -> on-node STRICT_PACK
        (2, 1, 2, True),  # TP2 x DP2
    ],
)
def test_ray_uni_backend_gate(tp, pp, dp, expected):
    assert (
        use_per_engine_strict_pack_pg(
            use_hybrid_engine=False,
            use_mp_backend=False,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
        )
        is expected
    )


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_mp_backend_never_per_engine_strict_pack(tp, pp):
    # The mp executor uses one node-atomic {GPU:tp_pp_size} bundle per engine,
    # so it never needs (and must not use) per-engine STRICT_PACK.
    assert not use_per_engine_strict_pack_pg(
        use_hybrid_engine=False,
        use_mp_backend=True,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=1,
    )


@pytest.mark.parametrize("tp,pp", [(1, 1), (2, 1), (4, 1), (2, 2)])
def test_hybrid_engine_never_per_engine_strict_pack(tp, pp):
    # colocate_all (hybrid) passes its own shared colocate PG; the per-engine
    # path must never engage.
    assert not use_per_engine_strict_pack_pg(
        use_hybrid_engine=True,
        use_mp_backend=False,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        data_parallel_size=1,
    )


@pytest.mark.parametrize(
    "reordered,tensor_pipeline_size,gpus_per_node,expected",
    [
        (list(range(32)), 4, 8, [list(range(start, start + 4)) for start in range(0, 32, 4)]),
        ([5, 2, 7, 0, 6, 3, 4, 1], 2, 4, [[5, 2], [7, 0], [6, 3], [4, 1]]),
    ],
)
def test_colocated_tp_groups_follow_node_order(reordered, tensor_pipeline_size, gpus_per_node, expected):
    layouts = [
        colocated_engine_bundle_indices(
            reordered_bundle_indices=reordered,
            engine_index=engine_index,
            data_parallel_rank=0,
            tensor_pipeline_size=tensor_pipeline_size,
            data_parallel_size=1,
            gpus_per_node=gpus_per_node,
        )
        for engine_index in range(len(reordered) // tensor_pipeline_size)
    ]

    assert layouts == expected


@pytest.mark.parametrize(
    "tensor_pipeline_size,error",
    [
        (16, "cannot fit on one 8-GPU policy node"),
        (3, "does not divide a 8-GPU policy node"),
    ],
)
def test_colocated_tp_group_invalid_node_geometry_is_rejected(tensor_pipeline_size, error):
    with pytest.raises(ValueError, match=error):
        colocated_engine_bundle_indices(
            reordered_bundle_indices=list(range(8)),
            engine_index=0,
            data_parallel_rank=0,
            tensor_pipeline_size=tensor_pipeline_size,
            data_parallel_size=1,
            gpus_per_node=8,
        )


def test_colocated_config_rejects_non_node_atomic_tp_geometry():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 24
    cfg.trainer.policy_mini_batch_size = 24
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.placement.policy_num_nodes = 3
    cfg.trainer.placement.policy_num_gpus_per_node = 8
    cfg.generator.num_inference_engines = 8
    cfg.generator.inference_engine_tensor_parallel_size = 3

    with pytest.raises(ValueError, match="does not divide a 8-GPU policy node"):
        validate_cfg(cfg)


def test_colocated_config_rejects_asynchronous_rollouts_before_allocation():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.placement.colocate_all = True
    cfg.trainer.rollout_buffer.max_staleness_steps = 1
    cfg.trainer.algorithm.policy_loss_type = "behavior_clip"

    with pytest.raises(ValueError, match="colocate_all requires"):
        validate_cfg(cfg)


@pytest.fixture
def inference_scheduler(monkeypatch):
    """Fake the Ray and vLLM boundary: record allocated bundles and answer each actor's placement report."""
    groups, actors, killed, removed = [], [], [], []
    report_changes = {}
    allocated = [0, 0]

    def placement_group(bundles, strategy):
        nodes, gpu_indices = {}, {}
        for index, bundle in enumerate(bundles):
            node = next(i for i, used in enumerate(allocated) if used + bundle["GPU"] <= 8)
            nodes[index] = f"node-{node}"
            gpu_indices[index] = allocated[node]
            allocated[node] += bundle["GPU"]
        pg = SimpleNamespace(
            bundle_specs=bundles,
            strategy=strategy,
            nodes=nodes,
            gpu_indices=gpu_indices,
            ready=lambda: None,
        )
        groups.append(pg)
        return pg

    class ActorClass:
        @staticmethod
        def options(**options):
            def remote(**kwargs):
                schedule = options["scheduling_strategy"]
                index = schedule.placement_group_bundle_index
                node = schedule.placement_group.nodes[index]
                rank = kwargs.get("data_parallel_rank", 0)
                size = kwargs.get("data_parallel_size", 1)
                # vLLM forms an EP group only when expert parallelism is on.
                ep_rank, ep_size = (rank, size) if kwargs.get("enable_expert_parallel") else (0, 1)
                report = InferenceWorkerPlacement(
                    node.replace("node", "host"),
                    f"GPU-{node}-{schedule.placement_group.gpu_indices[index]}",
                    rank,
                    size,
                    ep_rank,
                    ep_size,
                    rank,
                    size,
                )
                report = replace(report, **report_changes.get(len(actors), {}))
                actor = SimpleNamespace(
                    options=options,
                    kwargs=kwargs,
                    # The real reply crosses vLLM's msgpack utility RPC; keep that round trip.
                    report_engine_placement=SimpleNamespace(
                        remote=lambda: msgpack.unpackb(msgpack.packb([asdict(report)]), raw=False)
                    ),
                    report_engine_hosts=SimpleNamespace(remote=lambda: [report.host]),
                    get_model_max_len=SimpleNamespace(remote=lambda: 4096),
                    initialize_worker_numa_affinity=SimpleNamespace(remote=lambda: None),
                )
                actors.append(actor)
                return actor

            return SimpleNamespace(remote=remote)

    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="dev"))
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.models",
        SimpleNamespace(ModelRegistry=SimpleNamespace(get_supported_archs=lambda: [])),
    )
    monkeypatch.setitem(
        sys.modules,
        "skyrl_train.inference_engines.vllm.vllm_engine",
        SimpleNamespace(VLLMRayActor=ActorClass, AsyncVLLMRayActor=ActorClass),
    )
    monkeypatch.setattr(factory.AutoConfig, "from_pretrained", lambda *a, **kw: SimpleNamespace(model_type="test"))
    monkeypatch.setattr(factory, "placement_group", placement_group)
    monkeypatch.setattr(factory, "remove_placement_group", removed.append)
    monkeypatch.setattr(
        "skyrl_train.inference_engines.placement.placement_group_table", lambda pg: {"bundles_to_node_id": pg.nodes}
    )
    monkeypatch.setattr(factory.ray, "get", lambda ref, **kw: ref)
    monkeypatch.setattr(factory.ray, "wait", lambda refs, **kw: (refs, []))
    monkeypatch.setattr(factory.ray, "kill", killed.append)
    monkeypatch.setattr(
        factory.ray,
        "nodes",
        lambda: [
            {"Alive": True, "NodeID": f"node-{i}", "NodeManagerHostname": f"host-{i}", "Resources": {"GPU": 8}}
            for i in range(2)
        ],
    )
    monkeypatch.setattr(factory, "get_all_env_variables", SimpleNamespace(remote=lambda: {}))
    # The rendezvous helpers run Ray tasks and open sockets on the bundle's node.
    monkeypatch.setattr(factory, "get_pg_bundle_node_ips", lambda pg, indices: [pg.nodes[i] for i in indices])

    def reserve(pg, index, port_count, excluded_ports=()):
        base = 32000 + len(excluded_ports)
        return ReservedRendezvousPorts(pg.nodes[index], tuple(range(base, base + port_count)), object())

    monkeypatch.setattr(factory, "reserve_rendezvous_ports", reserve)

    def launch(**kwargs):
        return factory.create_ray_wrapped_inference_engines(
            num_inference_engines=kwargs.pop("num_inference_engines", 2),
            tensor_parallel_size=kwargs.pop("tensor_parallel_size", 1),
            pipeline_parallel_size=kwargs.pop("pipeline_parallel_size", 1),
            model_dtype="bfloat16",
            pretrain="test",
            seed=7,
            vllm_v1_disable_multiproc=True,
            enable_prefix_caching=True,
            enforce_eager=False,
            engine_init_timeout_seconds=30,
            engine_init_kwargs={"language_model_only": False},
            **kwargs,
        )

    return SimpleNamespace(
        launch=launch, groups=groups, actors=actors, killed=killed, removed=removed, report_changes=report_changes
    )


@pytest.mark.parametrize(
    "replicas,dp,expected_nodes",
    [
        (2, 8, ["node-0"] * 8 + ["node-1"] * 8),
        (2, 4, ["node-0"] * 8),
        (1, 16, ["node-0"] * 8 + ["node-1"] * 8),
    ],
)
def test_ep_replicas_pack_and_verify_workers_through_normal_config(inference_scheduler, replicas, dp, expected_nodes):
    scheduler = inference_scheduler
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.update(
        num_inference_engines=replicas,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=dp,
        inference_engine_expert_parallel_size=dp,
    )
    cfg.generator.engine_init_kwargs = {"language_model_only": False}
    engines = create_ray_wrapped_inference_engines_from_config(cfg, None, None)
    total = replicas * dp
    assert len(engines) == total
    assert [pg.strategy for pg in scheduler.groups] == ["PACK"] * replicas
    assert [sum(bundle["GPU"] for bundle in pg.bundle_specs) for pg in scheduler.groups] == [dp] * replicas
    assert [engine.weight_sync_relative_rank_offset for engine in engines] == [i // dp * dp for i in range(total)]
    placements = [placement for engine in engines for placement in engine.worker_placements]
    assert [row.node_id for row in placements] == expected_nodes
    assert [row.weight_receiver_rank for row in placements] == list(range(1, total + 1))
    assert [row.replica for row in placements] == [i // dp for i in range(total)]
    assert [row.bundle_index for row in placements] == list(range(dp)) * replicas
    assert [row.worker.dp_rank for row in placements] == list(range(dp)) * replicas
    assert [row.worker.ep_rank for row in placements] == list(range(dp)) * replicas
    assert {row.worker.ep_world_size for row in placements} == {dp}
    assert len({row.worker.gpu_uuid for row in placements}) == total


def test_data_parallel_workers_without_expert_parallelism_are_checked_with_no_ep_group(inference_scheduler):
    engines = inference_scheduler.launch(data_parallel_size=8, expert_parallel_size=1)
    placements = [placement for engine in engines for placement in engine.worker_placements]
    assert len(placements) == 16
    assert [pg.strategy for pg in inference_scheduler.groups] == ["STRICT_PACK"] * 2
    assert [row.node_id for row in placements] == ["node-0"] * 8 + ["node-1"] * 8
    assert {row.worker.ep_world_size for row in placements} == {1}


@pytest.mark.parametrize(
    "replicas,dp,tp,pp,strategy",
    [
        # Single-GPU engines share the flat group.
        (16, 1, 1, 1, "PACK"),
        # Tensor-parallel engines keep their own groups; their workers are not checked.
        (2, 1, 4, 1, "STRICT_PACK"),
        (2, 1, 1, 4, "STRICT_PACK"),
    ],
)
def test_engines_that_are_not_tp1_data_parallel_replicas_are_not_checked(
    inference_scheduler, replicas, dp, tp, pp, strategy
):
    engines = inference_scheduler.launch(
        num_inference_engines=replicas, data_parallel_size=dp, tensor_parallel_size=tp, pipeline_parallel_size=pp
    )
    assert {pg.strategy for pg in inference_scheduler.groups} == {strategy}
    assert all(engine.worker_placements is None for engine in engines)


def test_wrong_worker_topology_kills_the_replica_gang(inference_scheduler):
    scheduler = inference_scheduler
    scheduler.report_changes[1] = {"gpu_uuid": "GPU-node-0-0"}
    with pytest.raises(ValueError, match="distinct"):
        scheduler.launch(data_parallel_size=8, expert_parallel_size=8)
    assert scheduler.killed == scheduler.actors
    assert len(scheduler.killed) == 16
    assert scheduler.removed == scheduler.groups


@pytest.mark.parametrize(
    "change,error", [({"host": "wrong-host"}, "worker host"), ({"gpu_uuid": "GPU-node-0-0"}, "distinct")]
)
def test_cross_node_ep_rejects_wrong_worker_placement(inference_scheduler, change, error):
    scheduler = inference_scheduler
    scheduler.report_changes[8] = change
    with pytest.raises(ValueError, match=error):
        scheduler.launch(num_inference_engines=1, data_parallel_size=16, expert_parallel_size=16)
    assert len(scheduler.killed) == 16
    assert scheduler.removed == scheduler.groups


def _replica_reports():
    return [
        [asdict(InferenceWorkerPlacement(f"host-{replica}", f"GPU-{replica}-{rank}", rank, 8, rank, 8, rank, 8))]
        for replica in range(2)
        for rank in range(8)
    ]


def _verified_replicas(reports, offsets=None):
    return verified_inference_replica_placements(
        reports,
        bundle_nodes=[["node-0"] * 8, ["node-1"] * 8],
        node_hosts={"node-0": "host-0", "node-1": "host-1"},
        relative_rank_offsets=offsets if offsets is not None else [0] * 8 + [8] * 8,
        data_parallel_size=8,
        expert_parallel_size=8,
    )


def test_two_ep8_replicas_have_disjoint_gpus_and_receiver_ranks():
    placements = _verified_replicas(_replica_reports())
    assert [row.weight_receiver_rank for row in placements] == list(range(1, 17))
    assert {row.node_id for row in placements[:8]} == {"node-0"}
    assert {row.node_id for row in placements[8:]} == {"node-1"}
    assert {row.worker.gpu_uuid for row in placements[:8]}.isdisjoint(row.worker.gpu_uuid for row in placements[8:])


@pytest.mark.parametrize(
    "change,error",
    [
        ({"host": "host-other"}, "worker host"),
        ({"gpu_uuid": "GPU-0-0"}, "distinct"),
        ({"gpu_uuid": ""}, "distinct"),
        ({"ep_world_size": 16}, "EP rank or world size"),
        ({"dp_rank": 0}, "worker ranks"),
        ({"torch_world_size": 16}, "DP/torch world size"),
        ({"pp_rank": 1}, "PP rank or world size|placement bundles"),
    ],
)
def test_replica_topology_rejects_a_worker_that_disagrees_with_its_bundle(change, error):
    reports = _replica_reports()
    reports[1][0].update(change)
    with pytest.raises(ValueError, match=error):
        _verified_replicas(reports)


def test_a_dense_model_that_reports_no_expert_parallel_group_stays_node_local():
    # vLLM builds an EP group only for MoE models, even when EP=DP is requested.
    reports = _replica_reports()
    for report in reports:
        report[0].update(ep_rank=0, ep_world_size=1)
    assert len(_verified_replicas(reports)) == 16
    with pytest.raises(ValueError, match="spans nodes"):
        verified_inference_replica_placements(
            reports,
            bundle_nodes=[["node-0"] * 4 + ["node-1"] * 4] * 2,
            node_hosts={"node-0": "host-0", "node-1": "host-1"},
            relative_rank_offsets=[0] * 8 + [8] * 8,
            data_parallel_size=8,
            expert_parallel_size=8,
        )
    reports = _replica_reports()
    reports[3][0].update(ep_rank=0, ep_world_size=1)
    with pytest.raises(ValueError, match="EP rank or world size"):
        _verified_replicas(reports)


def test_replica_topology_rejects_reused_weight_receiver_ranks():
    with pytest.raises(ValueError, match="weight receiver ranks"):
        _verified_replicas(_replica_reports(), offsets=[0] * 16)


def test_replica_topology_rejects_a_missing_worker():
    with pytest.raises(ValueError, match="Incomplete"):
        _verified_replicas(_replica_reports()[:-1])


def test_two_stage_replicas_are_verified_per_stage():
    # One replica of DP=2 x PP=2: workers (dp, pp) at bundle dp*2+pp, stage 0 on node a, stage 1 on node b.
    reports = [
        [
            asdict(InferenceWorkerPlacement(f"host-{pp}", f"GPU-{dp}-{pp}", dp, 2, dp, 2, dp * 2 + pp, 4, pp, 2))
            for pp in range(2)
        ]
        for dp in range(2)
    ]
    placements = verified_inference_replica_placements(
        reports,
        bundle_nodes=[["node-0", "node-1"] * 2],
        node_hosts={"node-0": "host-0", "node-1": "host-1"},
        relative_rank_offsets=[0, 0],
        data_parallel_size=2,
        expert_parallel_size=2,
        pipeline_parallel_size=2,
    )
    assert [row.bundle_index for row in placements] == [0, 1, 2, 3]
    assert [row.weight_receiver_rank for row in placements] == [1, 2, 3, 4]
    reports[1][1]["host"] = "host-0"
    with pytest.raises(ValueError, match="stage 1 spans nodes"):
        verified_inference_replica_placements(
            reports,
            bundle_nodes=[["node-0", "node-1"] * 2],
            node_hosts={"node-0": "host-0", "node-1": "host-1"},
            relative_rank_offsets=[0, 0],
            data_parallel_size=2,
            expert_parallel_size=2,
            pipeline_parallel_size=2,
        )


@pytest.mark.parametrize("nodes", [{0: "a", 1: "b"}, {0: "a"}])
def test_node_local_bundles_must_be_complete_and_on_one_node(monkeypatch, nodes):
    monkeypatch.setattr(
        "skyrl_train.inference_engines.placement.placement_group_table",
        lambda pg: {"bundles_to_node_id": nodes},
    )
    with pytest.raises(ValueError, match="placement|bundles"):
        inference_bundle_nodes([object()], data_parallel_size=2, node_gpu_capacities={"a": 8, "b": 8})


def test_two_full_node_replicas_cannot_share_one_eight_gpu_node(monkeypatch):
    monkeypatch.setattr(
        "skyrl_train.inference_engines.placement.placement_group_table",
        lambda pg: {"bundles_to_node_id": {i: "node-0" for i in range(8)}},
    )
    with pytest.raises(ValueError, match="exceed GPU capacity"):
        inference_bundle_nodes([object(), object()], data_parallel_size=8, node_gpu_capacities={"node-0": 8})
