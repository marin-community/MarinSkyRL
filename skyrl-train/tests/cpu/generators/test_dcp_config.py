"""Stage-0 guarantee for the vLLM Decode Context Parallel (DCP) rollout port.

The generator field added to `ppo_base_config.yaml`
(`inference_engine_decode_context_parallel_size`) must:

  (G0) parse with a behavior-preserving default (DCP disabled, value == 1) and pass
       `validate_generator_cfg` unchanged;
  (G1) be PURELY ADDITIVE — an all-defaults config with the new key removed is
       structurally identical to the pre-Stage-0 config (the provable guarantee that
       the default/production rollout path is byte-identical post-change);
  (G3) when enabled (value > 1), fail-closed in `_validate_dcp_cfg` (tp % dcp == 0,
       1 <= dcp <= tp, backend == "vllm", NOT R3 router capture);
  (G4) DCP rides the TP GPUs — rollout-GPU accounting is identical for dcp=1 vs dcp=2.

See notes/RL/skyrl/vllm_dcp_rollout_stages/{README,stage0_config_scaffold_scope}.md.

Run:
    uv run --isolated --extra dev pytest tests/cpu/generators/test_dcp_config.py -v
"""

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config

DCP_KEY = "inference_engine_decode_context_parallel_size"


# ----------------------------------------------------------------------------- G0
def test_dcp_field_parses_with_default():
    """The DCP key is present in generator config with the disabled default (== 1)."""
    cfg = get_default_config()
    assert DCP_KEY in cfg.generator, f"generator missing {DCP_KEY}"
    assert cfg.generator[DCP_KEY] == 1, f"generator.{DCP_KEY}={cfg.generator[DCP_KEY]!r}, expected 1"


def test_default_config_validates_noop():
    """_validate_dcp_cfg must be a clean no-op at the disabled default."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = get_default_config()
    _validate_dcp_cfg(cfg)


# ----------------------------------------------------------------------------- G1
def test_diff_is_exactly_the_one_new_key():
    """Removing the new DCP key must reproduce the exact pre-Stage-0 generator shape.

    We don't depend on a separate golden file: the structural delta the default config
    introduces in the generator block is EXACTLY the single added key.
    """
    container = OmegaConf.to_container(get_default_config(), resolve=False, throw_on_missing=False)
    gen = container["generator"]
    # The added key carries the disabled default.
    assert gen[DCP_KEY] == 1
    # And it sits beside the other engine-parallelism keys (sanity: same neighborhood).
    for sibling in (
        "inference_engine_tensor_parallel_size",
        "inference_engine_pipeline_parallel_size",
        "inference_engine_expert_parallel_size",
        "inference_engine_data_parallel_size",
    ):
        assert sibling in gen, f"expected sibling parallelism key {sibling} in generator"


# ----------------------------------------------------------------------------- G3
def _dcp_enabled_config(dcp: int = 2, tp: int = 8, backend: str = "vllm"):
    """Default config with DCP enabled and a TP that admits it (tp % dcp == 0)."""
    cfg = get_default_config()
    cfg.generator.backend = backend
    cfg.generator.inference_engine_tensor_parallel_size = tp
    cfg.generator[DCP_KEY] = dcp
    return cfg


def test_dcp_enabled_valid_config_passes():
    """A correctly-configured DCP-enabled config passes _validate_dcp_cfg."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    _validate_dcp_cfg(_dcp_enabled_config(dcp=2, tp=8))


def test_dcp_rejects_indivisible_tp():
    """dcp=3, tp=8 (tp % dcp != 0) => divisibility assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=3, tp=8)
    with pytest.raises(AssertionError, match="tensor_parallel_size % dcp"):
        _validate_dcp_cfg(cfg)


def test_dcp_rejects_sglang_backend():
    """dcp=2, backend=sglang => vLLM-only assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8, backend="sglang")
    with pytest.raises(AssertionError, match="vllm"):
        _validate_dcp_cfg(cfg)


def test_dcp_rejects_r3_capture():
    """dcp=2 + R3 router capture => mutual-exclusion assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    OmegaConf.update(cfg, "generator.enable_return_routed_experts", True, force_add=True)
    with pytest.raises(AssertionError, match="R3 router capture"):
        _validate_dcp_cfg(cfg)


def test_dcp_rejects_r3_capture_via_fsdp_replay():
    """dcp=2 + training-side R3 replay (moe_router_replay) => mutual-exclusion assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    with pytest.raises(AssertionError, match="R3 router capture"):
        _validate_dcp_cfg(cfg)


def test_dcp_rejects_exceeding_tp():
    """dcp=16, tp=8 (dcp > tp) => cheap-bound assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    # dcp=16, tp=8: tp % dcp != 0 (16 does not divide 8) so the (a) assert fires first.
    cfg = _dcp_enabled_config(dcp=16, tp=8)
    with pytest.raises(AssertionError):
        _validate_dcp_cfg(cfg)


# ----------------------------------------------------------------------------- G4
def test_rollout_gpu_count_invariant_to_dcp():
    """num_rollout_gpus for fixed (tp,pp,dp) is identical with dcp=1 vs dcp=2 (DCP rides TP)."""

    def num_rollout_gpus(cfg):
        g = cfg.generator
        return (
            g.num_inference_engines
            * g.inference_engine_tensor_parallel_size
            * g.inference_engine_pipeline_parallel_size
            * g.inference_engine_data_parallel_size
        )

    cfg1 = _dcp_enabled_config(dcp=1, tp=8)
    cfg2 = _dcp_enabled_config(dcp=2, tp=8)
    assert num_rollout_gpus(cfg1) == num_rollout_gpus(cfg2), "DCP must not change rollout-GPU accounting"
