"""Unit tests for inference-engine placement and startup configuration.

The placement strategy checks verify that the ray/uni backend chooses:
  - per-engine STRICT_PACK ONLY for multi-GPU engines (TP*PP*DP > 1), to keep each
    engine's TP/PP workers and DP ranks on one node (#232 cross-node all-reduce fix), and
  - the flat PACK fallback for single-GPU engines (TP==PP==1), so single-GPU
    bundles pack densely and leave whole nodes free for the downstream policy
    PACK PG (the lever1/swesmith multi-node starvation regression fix), and
  - never per-engine STRICT_PACK on the hybrid (colocate_all) or mp-backend
    paths (the mp {GPU:tp_pp_size} bundle is already node-atomic).

uv run --isolated --group dev --extra cpu pytest tests/cpu/test_engine_placement_strategy.py
"""

import pytest

from omegaconf import OmegaConf

from skyrl_train.inference_engines.ray_wrapped_inference_engine import (
    assert_data_parallel_ranks_share_a_node,
    resolve_engine_max_model_len,
)
from skyrl_train.utils.placement_geometry import (
    EnginePlacementLayout,
    colocated_engine_bundle_indices,
    data_parallel_rank_bundle_indices,
)
from skyrl_train.utils.utils import use_per_engine_strict_pack_pg, validate_cfg
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
        (1, 1, 1, False),  # lever1 (16 TP=1 engines) / swesmith (48) -> flat PACK, dense
        (2, 1, 1, True),  # de-risk geometry on ray/uni -> on-node STRICT_PACK
        (4, 1, 1, True),  # #232 TP=4 on 4-GPU nodes -> on-node STRICT_PACK (this is the bug it fixed)
        (1, 2, 1, True),  # PP=2 single TP -> multi-GPU engine, still needs on-node
        (2, 2, 1, True),  # TP*PP=4
        (1, 1, 4, True),  # DP4xEP4 on 4-GPU nodes -> one STRICT_PACK PG per engine
        (1, 1, 8, True),  # DP8xEP8 on 8-GPU nodes -> STRICT_PACK
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
    "layout,engine_index,expected",
    [
        (EnginePlacementLayout.HYBRID, 1, [4, 6]),  # engine 1's two DP replicas: bundles [4,5] and [6,7]
        (EnginePlacementLayout.PER_ENGINE, 1, [0, 2]),  # engine-local indices, one TP*PP slice per rank
        (EnginePlacementLayout.MP, 1, [2, 3]),  # one {GPU: tp*pp} bundle per (engine, rank)
    ],
)
def test_data_parallel_rank_bundles_follow_the_placement_layout(layout, engine_index, expected):
    colocated = [[0, 1], [2, 3], [4, 5], [6, 7]]

    assert (
        data_parallel_rank_bundle_indices(
            layout,
            engine_index=engine_index,
            data_parallel_size=2,
            tensor_pipeline_size=2,
            colocated_engine_bundles=colocated,
        )
        == expected
    )


@pytest.mark.parametrize("node_ips,fails", [(["10.0.0.1", "10.0.0.2"], True), (["10.0.0.1", "10.0.0.1"], False)])
def test_data_parallel_ranks_must_resolve_to_one_node(node_ips, fails):
    probe = lambda placement_group, bundle_indices: node_ips  # noqa: E731

    if fails:
        with pytest.raises(RuntimeError, match="must be node-local"):
            assert_data_parallel_ranks_share_a_node(None, [0, 1], engine_index=0, node_ips_of=probe)
    else:
        assert_data_parallel_ranks_share_a_node(None, [0, 1], engine_index=0, node_ips_of=probe)


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


def _validatable_config():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    return cfg


def test_config_rejects_nonpositive_engine_startup_timeout():
    cfg = _validatable_config()
    cfg.generator.engine_init_timeout_seconds = 0

    with pytest.raises(ValueError, match="engine_init_timeout_seconds must be greater than zero"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "key,value,error",
    [
        ("pause_mode", "wait", "trainer.fully_async.pause_mode must be one of"),
        ("clear_kv_cache_on_weight_sync", "yes", "clear_kv_cache_on_weight_sync must be boolean"),
        ("first_token_admission", 1, "first_token_admission must be boolean"),
        ("max_buffered_groups", 0, "max_buffered_groups must be a positive integer or null"),
    ],
)
def test_config_rejects_malformed_weight_sync_and_buffer_settings(key, value, error):
    cfg = _validatable_config()
    OmegaConf.update(cfg, f"trainer.fully_async.{key}", value)

    with pytest.raises(ValueError, match=error):
        validate_cfg(cfg)
