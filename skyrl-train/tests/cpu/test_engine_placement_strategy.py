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

from skyrl_train.inference_engines.placement import colocated_engine_bundle_indices
from skyrl_train.utils.utils import validate_cfg
from skyrl_train.utils.utils import (
    use_per_engine_strict_pack_pg,
)
from tests.cpu.util import example_dummy_config


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


def test_colocated_tp_groups_cannot_span_node_bundles():
    reordered = list(range(32))
    layouts = [
        colocated_engine_bundle_indices(
            reordered_bundle_indices=reordered,
            engine_index=engine_index,
            data_parallel_rank=0,
            tensor_pipeline_size=4,
            data_parallel_size=1,
            gpus_per_node=8,
        )
        for engine_index in range(8)
    ]

    assert layouts == [list(range(start, start + 4)) for start in range(0, 32, 4)]


def test_colocated_tp_groups_use_ray_node_order_not_original_bundle_order():
    reordered = [5, 2, 7, 0, 6, 3, 4, 1]

    layouts = [
        colocated_engine_bundle_indices(
            reordered_bundle_indices=reordered,
            engine_index=engine_index,
            data_parallel_rank=0,
            tensor_pipeline_size=2,
            data_parallel_size=1,
            gpus_per_node=4,
        )
        for engine_index in range(4)
    ]

    assert layouts == [[5, 2], [7, 0], [6, 3], [4, 1]]


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

    with pytest.raises(AssertionError, match="does not divide a 8-GPU policy node"):
        validate_cfg(cfg)


def test_config_rejects_nonpositive_engine_startup_timeout():
    cfg = example_dummy_config()
    cfg.trainer.train_batch_size = 4
    cfg.trainer.policy_mini_batch_size = 4
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.generator.engine_init_timeout_seconds = 0

    with pytest.raises(ValueError, match="engine_init_timeout_seconds must be greater than zero"):
        validate_cfg(cfg)
