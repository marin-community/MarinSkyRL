"""
Unit tests for the inference-engine placement-group strategy gate
(``use_per_engine_strict_pack_pg``).

Pure (Ray-free) checks that the ray/uni backend chooses:
  - per-engine STRICT_PACK ONLY for multi-GPU engines (TP*PP > 1), to keep each
    engine's TP/PP workers on one node (#232 cross-node all-reduce fix), and
  - the flat PACK fallback for single-GPU engines (TP==PP==1), so single-GPU
    bundles pack densely and leave whole nodes free for the downstream policy
    PACK PG (the lever1/swesmith multi-node starvation regression fix), and
  - never per-engine STRICT_PACK on the hybrid (colocate_all) or mp-backend
    paths (the mp {GPU:tp_pp_size} bundle is already node-atomic).

uv run --isolated --extra dev pytest tests/cpu/test_engine_placement_strategy.py
"""

import pytest

from skyrl_train.utils.utils import use_per_engine_strict_pack_pg


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
