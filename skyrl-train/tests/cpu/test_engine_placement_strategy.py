"""Unit tests for inference-engine placement and startup configuration.

The placement strategy checks verify that the ray/uni backend chooses:
  - per-engine STRICT_PACK ONLY for multi-GPU engines (TP*PP > 1), to keep each
    engine's TP/PP workers on one node (#232 cross-node all-reduce fix), and
  - the flat PACK fallback for single-GPU engines (TP==PP==1), so single-GPU
    bundles pack densely and leave whole nodes free for the downstream policy
    PACK PG (the lever1/swesmith multi-node starvation regression fix), and
  - never per-engine STRICT_PACK on the hybrid (colocate_all) or mp-backend
    paths (the mp {GPU:tp_pp_size} bundle is already node-atomic).

uv run --isolated --group dev --extra cpu pytest tests/cpu/test_engine_placement_strategy.py
"""

import pytest
from dataclasses import asdict, replace
from types import SimpleNamespace
import sys
import msgpack

from marinskyrl.inference_placement import InferenceWorkerPlacement, validate_node_local_config
from skyrl_train.inference_engines.placement import (
    inference_worker_placement,
    node_local_bundle_nodes,
    verified_inference_replica_placements,
)
from skyrl_train.entrypoints.main_base import create_ray_wrapped_inference_engines_from_config

from skyrl_train.utils.placement_geometry import colocated_engine_bundle_indices
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.utils.utils import (
    use_per_engine_strict_pack_pg,
)
from tests.cpu.util import example_dummy_config


@pytest.fixture
def inference_scheduler(monkeypatch):
    """Ray/vLLM boundary fake that records allocated bundles and actor observations."""
    import skyrl_train.inference_engines.ray_wrapped_inference_engine as factory

    groups, actors, events, killed, removed = [], [], [], [], []
    diagnostic_changes = {}

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
                report = replace(report, **diagnostic_changes.get(len(actors), {}))
                actor = SimpleNamespace(
                    options=options,
                    kwargs=kwargs,
                    report_engine_placement=SimpleNamespace(
                        remote=lambda: msgpack.unpackb(msgpack.packb([asdict(report)]), raw=False)
                    ),
                    report_engine_hosts=SimpleNamespace(remote=lambda: [report.host]),
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
    monkeypatch.setattr(factory, "record_event", lambda name, fields, **kw: events.append((name, fields, kw)))

    def launch(**kwargs):
        return factory.create_ray_wrapped_inference_engines(
            num_inference_engines=kwargs.pop("num_inference_engines", 2),
            tensor_parallel_size=1,
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
        launch=launch,
        groups=groups,
        actors=actors,
        events=events,
        killed=killed,
        removed=removed,
        diagnostic_changes=diagnostic_changes,
    )


def test_node_local_factory_allocates_two_ep8_replicas_and_publishes_verified_topology(inference_scheduler):
    scheduler = inference_scheduler
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.update(
        num_inference_engines=2,
        inference_engine_tensor_parallel_size=1,
        inference_engine_pipeline_parallel_size=1,
        inference_engine_data_parallel_size=8,
        inference_engine_expert_parallel_size=8,
        inference_engine_node_local=True,
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
    assert len(scheduler.events) == 16
    assert {name for name, _, _ in scheduler.events} == {"inference_replica_topology"}
    rows = [fields for _, fields, _ in scheduler.events]
    assert [row["expected_weight_receiver_rank"] for row in rows] == list(range(1, 17))
    assert {row["ep_world_size"] for row in rows} == {8}
    assert len({row["gpu_uuid"] for row in rows}) == 16


@pytest.mark.parametrize("replicas,dp,ep", [(16, 1, 1), (1, 16, 16)])
def test_default_placement_preserves_single_gpu_packing_and_multinode_ep16(inference_scheduler, replicas, dp, ep):
    scheduler = inference_scheduler
    engines = scheduler.launch(num_inference_engines=replicas, data_parallel_size=dp, expert_parallel_size=ep)
    assert len(engines) == 16
    assert [pg.strategy for pg in scheduler.groups] == ["PACK"]
    assert len(scheduler.groups[0].bundle_specs) == 16
    assert scheduler.events == []


def test_node_local_oversized_replica_fails_before_gpu_allocation(inference_scheduler):
    scheduler = inference_scheduler
    with pytest.raises(ValueError, match="needs 16 GPUs"):
        scheduler.launch(node_local=True, data_parallel_size=16, expert_parallel_size=16)
    assert scheduler.groups == []
    assert scheduler.actors == []


def test_invalid_worker_topology_kills_replica_gang_without_publishing_verified_event(inference_scheduler):
    scheduler = inference_scheduler
    scheduler.diagnostic_changes[1] = {"gpu_uuid": "GPU-node-0-0"}
    with pytest.raises(ValueError, match="distinct"):
        scheduler.launch(node_local=True, data_parallel_size=8, expert_parallel_size=8)
    assert scheduler.killed == scheduler.actors
    assert len(scheduler.killed) == 16
    assert scheduler.removed == scheduler.groups
    assert scheduler.events == []


@pytest.mark.parametrize(
    "changes",
    [
        {"backend": "sglang"},
        {"async_engine": False},
        {"inference_engine_tensor_parallel_size": 2},
        {"inference_engine_pipeline_parallel_size": 2},
        {"run_engines_locally": False},
        {"inference_engine_data_parallel_size": 8, "inference_engine_expert_parallel_size": 1},
    ],
)
def test_node_local_config_rejects_unsupported_modes(changes):
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = False
    cfg.generator.inference_engine_node_local = True
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.async_engine = True
    cfg.generator.update(changes)
    with pytest.raises(ValueError, match="inference_engine_node_local"):
        validate_node_local_config(cfg)


def test_node_local_config_rejects_colocation():
    cfg = example_dummy_config()
    cfg.trainer.placement.colocate_all = True
    cfg.generator.inference_engine_node_local = True
    with pytest.raises(ValueError, match="non-colocated"):
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
        replica_nodes=["node-0", "node-1"],
        node_hosts={"node-0": "host-0", "node-1": "host-1"},
        relative_rank_offsets=offsets if offsets is not None else [0] * 8 + [8] * 8,
        data_parallel_size=8,
        expert_parallel_size=8,
    )


def test_two_ep8_replica_observations_have_disjoint_gpus_and_receiver_ranks():
    placements = _verified_replicas(_replica_reports())
    assert [row.expected_weight_receiver_rank for row in placements] == list(range(1, 17))
    assert {row.node_id for row in placements[:8]} == {"node-0"}
    assert {row.node_id for row in placements[8:]} == {"node-1"}
    assert {row.worker.gpu_uuid for row in placements[:8]}.isdisjoint(row.worker.gpu_uuid for row in placements[8:])


@pytest.mark.parametrize(
    "change,error",
    [
        ({"host": "host-other"}, "spans nodes"),
        ({"gpu_uuid": "GPU-0-0"}, "distinct"),
        ({"ep_world_size": 16}, "EP rank or world size"),
        ({"dp_rank": 0}, "worker ranks"),
        ({"torch_world_size": 16}, "DP/torch world size"),
        ({"gpu_uuid": ""}, "distinct"),
    ],
)
def test_replica_topology_rejects_wrong_actual_worker_mapping(change, error):
    reports = _replica_reports()
    reports[1][0].update(change)
    with pytest.raises(ValueError, match=error):
        _verified_replicas(reports)


def test_replica_topology_rejects_reused_weight_receiver_ranks():
    with pytest.raises(ValueError, match="weight receiver ranks"):
        _verified_replicas(_replica_reports(), offsets=[0] * 16)


def test_replica_topology_rejects_missing_worker():
    with pytest.raises(ValueError, match="Incomplete"):
        _verified_replicas(_replica_reports()[:-1])


@pytest.mark.parametrize("nodes", [{0: "a", 1: "b"}, {0: "a"}])
def test_node_local_placement_rejects_split_or_missing_bundles(monkeypatch, nodes):
    monkeypatch.setattr(
        "skyrl_train.inference_engines.placement.placement_group_table",
        lambda pg: {"bundles_to_node_id": nodes},
    )
    with pytest.raises(ValueError, match="placement|bundles"):
        node_local_bundle_nodes([object()], data_parallel_size=2, node_gpu_capacities={"a": 8, "b": 8})


def test_full_node_replicas_cannot_reuse_one_eight_gpu_node(monkeypatch):
    monkeypatch.setattr(
        "skyrl_train.inference_engines.placement.placement_group_table",
        lambda pg: {"bundles_to_node_id": {i: "node-0" for i in range(8)}},
    )
    with pytest.raises(ValueError, match="exceed GPU capacity"):
        node_local_bundle_nodes([object(), object()], data_parallel_size=8, node_gpu_capacities={"node-0": 8})


def test_worker_diagnostic_reads_actual_cuda_device_uuid(monkeypatch):
    monkeypatch.setattr("torch.cuda.current_device", lambda: 3)
    properties = {3: SimpleNamespace(uuid="GPU-actual-worker")}
    monkeypatch.setattr("torch.cuda.get_device_properties", properties.__getitem__)
    monkeypatch.setattr("torch.distributed.get_rank", lambda: 7)
    monkeypatch.setattr("torch.distributed.get_world_size", lambda: 8)
    report = inference_worker_placement(dp_rank=7, dp_world_size=8, ep_rank=7, ep_world_size=8)
    assert report.gpu_uuid == "GPU-actual-worker"
    assert report.torch_rank == 7
    assert report.torch_world_size == 8


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
