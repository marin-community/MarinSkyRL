"""Unit tests for inference-engine placement and startup configuration.

The placement strategy checks verify that the ray/uni backend chooses:
  - per-engine STRICT_PACK ONLY for multi-GPU engines (TP*PP > 1), to keep each
    engine's TP/PP workers on one node (#232 cross-node all-reduce fix), and
  - the flat PACK fallback for single-GPU engines (TP==PP==1), so single-GPU
    bundles pack densely and leave whole nodes free for the downstream policy
    PACK PG (the lever1/swesmith multi-node starvation regression fix), and
  - never per-engine STRICT_PACK on the hybrid (colocate_all) or mp-backend
    paths (the mp {GPU:tp_pp_size} bundle is already node-atomic), and
  - one group per node-local replica (generator.inference_engine_node_local), with its workers
    checked against the bundles they were given.

uv run --isolated --group dev --extra cpu pytest tests/cpu/test_engine_placement_strategy.py
"""

from dataclasses import asdict, replace
import sys
from types import SimpleNamespace

import msgpack
import pytest

from marinskyrl.inference_placement import InferenceWorkerPlacement, node_local_blocker, validate_node_local_config
from marinskyrl.runtime_options import NodeLocalPlacement
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config
from skyrl_train.inference_engines.placement import node_local_bundle_nodes, verified_inference_replica_placements
from skyrl_train.inference_engines import ray_wrapped_inference_engine as factory
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
    "tp,pp,expected",
    [
        (1, 1, False),  # lever1 (16 TP=1 engines) / swesmith (48) -> flat PACK, dense
        (2, 1, True),  # de-risk geometry on ray/uni -> on-node STRICT_PACK
        (4, 1, True),  # #232 TP=4 -> on-node STRICT_PACK (this is the bug it fixed)
        (1, 2, True),  # PP=2 single TP -> multi-GPU engine, still needs on-node
        (2, 2, True),  # TP*PP=4
    ],
)
def test_ray_uni_backend_gate(tp, pp, expected):
    assert (
        use_per_engine_strict_pack_pg(
            use_hybrid_engine=False,
            use_mp_backend=False,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
        )
        is expected
    )


def test_tp1_never_strict_pack_so_policy_pg_not_starved():
    # The exact lever1/swesmith regression: multi-node TP=1 must NOT use
    # per-engine STRICT_PACK (which scatters 1-GPU bundles and starves the
    # policy PACK PG of its whole nodes).
    assert not use_per_engine_strict_pack_pg(
        use_hybrid_engine=False,
        use_mp_backend=False,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
    )


def test_tp4_on_4gpu_node_still_strict_pack():
    # Guards against the WRONG `per_engine_gpu_count > gpus_per_node` gate:
    # TP=4 on 4-GPU nodes (4 is not > 4) must still use STRICT_PACK, else #232
    # (cross-node TP all-reduce decode deadlock) re-breaks.
    assert use_per_engine_strict_pack_pg(
        use_hybrid_engine=False,
        use_mp_backend=False,
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
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


def test_config_rejects_nonpositive_engine_startup_timeout():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.generator.engine_init_timeout_seconds = 0

    with pytest.raises(ValueError, match="engine_init_timeout_seconds must be greater than zero"):
        validate_cfg(cfg)


@pytest.fixture
def inference_scheduler(monkeypatch):
    """Fake the Ray and vLLM boundary: record allocated bundles and answer each actor's placement report."""
    groups, actors, killed, removed = [], [], [], []
    report_changes = {}

    def placement_group(bundles, strategy):
        index = len(groups)
        pg = SimpleNamespace(
            bundle_specs=bundles,
            strategy=strategy,
            nodes={i: f"node-{index if strategy == 'STRICT_PACK' else i // 8}" for i in range(len(bundles))},
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
                report = InferenceWorkerPlacement(
                    node.replace("node", "host"), f"GPU-{node}-{index}", rank, size, rank, size, rank, size
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
    # The rendezvous helper makes a Ray RPC and opens a socket on the selected node.
    monkeypatch.setattr(
        factory, "get_rendezvous_addr_port", lambda pg, index, ports: (pg.nodes[index], 32000 + len(ports))
    )

    def launch(**kwargs):
        return factory.create_ray_wrapped_inference_engines(
            num_inference_engines=kwargs.pop("num_inference_engines", 2),
            tensor_parallel_size=kwargs.pop("tensor_parallel_size", 1),
            pipeline_parallel_size=1,
            model_dtype="bfloat16",
            pretrain="test",
            seed=7,
            vllm_v1_disable_multiproc=True,
            enable_prefix_caching=True,
            enforce_eager=False,
            engine_init_timeout_seconds=30,
            async_engine=True,
            engine_init_kwargs={"language_model_only": False},
            **kwargs,
        )

    return SimpleNamespace(
        launch=launch, groups=groups, actors=actors, killed=killed, removed=removed, report_changes=report_changes
    )


def test_the_default_config_packs_two_ep8_replicas_on_a_node_each_and_verifies_their_workers(inference_scheduler):
    scheduler = inference_scheduler
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.update(
        num_inference_engines=2,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
        async_engine=True,
    )
    cfg.generator.engine_init_kwargs = {"language_model_only": False}
    engines = create_ray_wrapped_inference_engines_from_config(cfg, None, None)
    assert len(engines) == 16
    assert [pg.strategy for pg in scheduler.groups] == ["STRICT_PACK", "STRICT_PACK"]
    assert [sum(bundle["GPU"] for bundle in pg.bundle_specs) for pg in scheduler.groups] == [8, 8]
    assert [engine.weight_sync_relative_rank_offset for engine in engines] == [0] * 8 + [8] * 8
    endpoints = [
        (actor.kwargs["data_parallel_address"], actor.kwargs["data_parallel_rpc_port"]) for actor in scheduler.actors
    ]
    assert len(set(endpoints[:8])) == len(set(endpoints[8:])) == 1
    assert endpoints[0] != endpoints[8]
    assert all("node_local" not in actor.kwargs for actor in scheduler.actors)
    placements = [placement for engine in engines for placement in engine.worker_placements]
    assert [row.weight_receiver_rank for row in placements] == list(range(1, 17))
    assert [row.replica for row in placements] == [0] * 8 + [1] * 8
    assert {row.worker.ep_world_size for row in placements} == {8}
    assert len({row.worker.gpu_uuid for row in placements}) == 16


@pytest.mark.parametrize(
    "replicas,dp,ep,mode",
    [
        # Sixteen single-GPU engines: one-bundle STRICT_PACK groups would scatter and starve the policy group.
        (16, 1, 1, NodeLocalPlacement.AUTO),
        # One EP16 replica does not fit an 8-GPU node and spans two, as it always has.
        (1, 16, 16, NodeLocalPlacement.AUTO),
        # Data-parallel ranks without expert parallelism exchange nothing between them.
        (2, 8, 1, NodeLocalPlacement.AUTO),
        # Four-GPU replicas on two 8-GPU nodes: groups of their own could leave both nodes partly used.
        (4, 4, 4, NodeLocalPlacement.AUTO),
        (2, 8, 8, NodeLocalPlacement.OFF),
    ],
)
def test_engines_auto_does_not_pack_keep_the_flat_pack_group(inference_scheduler, replicas, dp, ep, mode):
    scheduler = inference_scheduler
    engines = scheduler.launch(
        num_inference_engines=replicas, data_parallel_size=dp, expert_parallel_size=ep, node_local_placement=mode
    )
    assert len(engines) == 16
    assert [pg.strategy for pg in scheduler.groups] == ["PACK"]
    assert len(scheduler.groups[0].bundle_specs) == 16
    assert all(engine.worker_placements is None for engine in engines)


def test_tensor_parallel_engines_keep_their_per_engine_groups_without_verification(inference_scheduler):
    scheduler = inference_scheduler
    engines = scheduler.launch(tensor_parallel_size=4)
    assert [pg.strategy for pg in scheduler.groups] == ["STRICT_PACK", "STRICT_PACK"]
    assert all(engine.worker_placements is None for engine in engines)


def test_require_packs_replicas_smaller_than_a_node(inference_scheduler):
    scheduler = inference_scheduler
    engines = scheduler.launch(
        data_parallel_size=4, expert_parallel_size=4, node_local_placement=NodeLocalPlacement.REQUIRE
    )
    assert [pg.strategy for pg in scheduler.groups] == ["STRICT_PACK", "STRICT_PACK"]
    assert [len(pg.bundle_specs) for pg in scheduler.groups] == [4, 4]
    assert all(len(engine.worker_placements) == 1 for engine in engines)


def test_require_refuses_an_oversized_replica_before_gpu_allocation(inference_scheduler):
    scheduler = inference_scheduler
    with pytest.raises(ValueError, match="needs 16 GPUs"):
        scheduler.launch(
            data_parallel_size=16, expert_parallel_size=16, node_local_placement=NodeLocalPlacement.REQUIRE
        )
    assert scheduler.groups == []
    assert scheduler.actors == []


def test_wrong_worker_topology_kills_the_replica_gang(inference_scheduler):
    scheduler = inference_scheduler
    scheduler.report_changes[1] = {"gpu_uuid": "GPU-node-0-0"}
    with pytest.raises(ValueError, match="distinct"):
        scheduler.launch(data_parallel_size=8, expert_parallel_size=8)
    assert scheduler.killed == scheduler.actors
    assert len(scheduler.killed) == 16
    assert scheduler.removed == scheduler.groups


NODE_LOCAL_ENGINE = dict(
    mode=NodeLocalPlacement.AUTO,
    backend="vllm",
    async_engine=True,
    colocated=False,
    remote=False,
    mp_executor=False,
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=8,
    expert_parallel_size=8,
    num_inference_engines=2,
    node_gpu_capacities=[8, 8, 8],
)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": NodeLocalPlacement.OFF},
        {"backend": "sglang"},
        {"async_engine": False},
        {"colocated": True},
        {"remote": True},
        {"mp_executor": True},
        {"tensor_parallel_size": 2},
        {"data_parallel_size": 1, "expert_parallel_size": 1},
        # Even on one node, where nothing can scatter, a single-GPU engine has no replica to pack.
        {"data_parallel_size": 1, "expert_parallel_size": 1, "node_gpu_capacities": [8]},
        {"expert_parallel_size": 1},
        # No node is large enough for a stage; too few nodes for every stage.
        {"node_gpu_capacities": [4, 4, 4, 4]},
        {"node_gpu_capacities": [8]},
        # Mixed nodes: the 4-GPU nodes cannot take an 8-GPU stage, and one 8-GPU node is not enough for two.
        {"node_gpu_capacities": [8, 4, 4]},
        # A 4-GPU engine on 8-GPU nodes fills half a node.
        {"data_parallel_size": 4, "expert_parallel_size": 4},
    ],
)
def test_auto_packs_only_an_engine_that_can_hold_a_replica_without_fragmenting_nodes(change):
    assert node_local_blocker(**NODE_LOCAL_ENGINE) is None
    assert node_local_blocker(**{**NODE_LOCAL_ENGINE, **change}) is not None


@pytest.mark.parametrize(
    "change",
    [
        # Half-node engines, packed on request or when the cluster is one node and nothing can scatter.
        {"mode": NodeLocalPlacement.REQUIRE, "data_parallel_size": 4, "expert_parallel_size": 4},
        {"data_parallel_size": 2, "expert_parallel_size": 2, "node_gpu_capacities": [8]},
        # Two stages of four GPUs fill one node.
        {"data_parallel_size": 4, "expert_parallel_size": 4, "pipeline_parallel_size": 2},
    ],
)
def test_engines_smaller_than_a_node_are_packed_when_nothing_can_scatter_or_on_request(change):
    assert node_local_blocker(**{**NODE_LOCAL_ENGINE, **change}) is None


def test_config_refuses_require_for_an_engine_that_can_never_be_node_local_and_an_unknown_mode():
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.update(
        async_engine=True,
        inference_engine_tensor_parallel_size=2,
        inference_engine_node_local="require",
    )
    with pytest.raises(ValueError, match="require cannot be honoured: it needs TP=1"):
        validate_node_local_config(cfg)
    # The same shape under the default mode is placed as before, so the config is accepted.
    cfg.generator.inference_engine_node_local = "auto"
    validate_node_local_config(cfg)
    cfg.generator.inference_engine_node_local = True
    with pytest.raises(ValueError, match="not a valid NodeLocalPlacement"):
        validate_node_local_config(cfg)


def _replica_reports():
    return [
        [asdict(InferenceWorkerPlacement(f"host-{replica}", f"GPU-{replica}-{rank}", rank, 8, rank, 8, rank, 8))]
        for replica in range(2)
        for rank in range(8)
    ]


def _verified_replicas(reports, offsets=None):
    return verified_inference_replica_placements(
        reports,
        stage_nodes=[["node-0"], ["node-1"]],
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
        ({"host": "host-other"}, "spans nodes"),
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


def test_a_dense_model_that_reports_no_expert_parallel_group_is_verified_not_killed():
    # vLLM builds an EP group only for MoE models: a dense model asked for EP=DP reports size 1 everywhere.
    reports = _replica_reports()
    for report in reports:
        report[0].update(ep_rank=0, ep_world_size=1)
    assert len(_verified_replicas(reports)) == 16
    # One worker of a MoE replica reporting no EP group is still a mismatch.
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
        stage_nodes=[["node-0", "node-1"]],
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
            stage_nodes=[["node-0", "node-1"]],
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
        node_local_bundle_nodes([object()], data_parallel_size=2, node_gpu_capacities={"a": 8, "b": 8})


def test_two_full_node_replicas_cannot_share_one_eight_gpu_node(monkeypatch):
    monkeypatch.setattr(
        "skyrl_train.inference_engines.placement.placement_group_table",
        lambda pg: {"bundles_to_node_id": {i: "node-0" for i in range(8)}},
    )
    with pytest.raises(ValueError, match="exceed GPU capacity"):
        node_local_bundle_nodes([object(), object()], data_parallel_size=8, node_gpu_capacities={"node-0": 8})
