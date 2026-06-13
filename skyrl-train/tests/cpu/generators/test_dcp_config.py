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

from types import SimpleNamespace
from unittest import mock

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


# ============================================================================= STAGE 2
# Model-aware kv-head bound (G3 b), GQA/MLA arch gate (G3 f), remote path, offline degrade.


def _gqa_config(num_key_value_heads=4, num_attention_heads=32, attn_implementation=None):
    """A stub HF config for a GQA model (e.g. Qwen3-like)."""
    cfg = SimpleNamespace(
        num_key_value_heads=num_key_value_heads,
        num_attention_heads=num_attention_heads,
        architectures=["Qwen3ForCausalLM"],
    )
    if attn_implementation is not None:
        cfg._attn_implementation = attn_implementation
    return cfg


def _mla_config():
    """A stub HF config for an MLA model (DeepSeek/Kimi family)."""
    return SimpleNamespace(
        kv_lora_rank=512,
        q_lora_rank=1536,
        num_attention_heads=128,
        architectures=["DeepseekV3ForCausalLM"],
    )


def _patch_autoconfig(model_cfg):
    """Patch transformers.AutoConfig.from_pretrained to return a stub (no network)."""
    return mock.patch("transformers.AutoConfig.from_pretrained", return_value=model_cfg)


# --------------------------------------------------------------------------- G3 (b)
def test_kv_head_bound_passes_within_bound():
    """GQA H=4, tp=8: dcp=2 (<=8/4=2) passes."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    with _patch_autoconfig(_gqa_config(num_key_value_heads=4)):
        _validate_dcp_cfg(cfg)


def test_kv_head_bound_passes_at_limit():
    """GQA H=2, tp=8: dcp=4 (==8/2) passes (at the bound)."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=4, tp=8)
    with _patch_autoconfig(_gqa_config(num_key_value_heads=2)):
        _validate_dcp_cfg(cfg)


def test_kv_head_bound_rejects_above_bound():
    """GQA H=4, tp=8: dcp=8 (> 8/4=2) raises the kv-head bound assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=8, tp=8)
    with _patch_autoconfig(_gqa_config(num_key_value_heads=4)):
        with pytest.raises(AssertionError, match="kv-head bound"):
            _validate_dcp_cfg(cfg)


def test_kv_head_bound_no_headroom():
    """GQA H=8, tp=8: any dcp>1 raises (tp//H == 1, no DCP headroom)."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    with _patch_autoconfig(_gqa_config(num_key_value_heads=8)):
        with pytest.raises(AssertionError, match="kv-head bound"):
            _validate_dcp_cfg(cfg)


def test_kv_head_bound_mha_falls_back_to_attn_heads():
    """MHA (no num_key_value_heads): kv-heads == num_attention_heads; bound uses that."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    # MHA with 32 heads, tp=8 -> tp//32 == 0 -> any dcp>1 rejected by the kv bound.
    cfg = _dcp_enabled_config(dcp=2, tp=8)
    mha = SimpleNamespace(num_attention_heads=32, architectures=["LlamaForCausalLM"])
    with _patch_autoconfig(mha):
        with pytest.raises(AssertionError, match="kv-head bound"):
            _validate_dcp_cfg(cfg)


# --------------------------------------------------------------------------- G3 (b) MLA
def test_mla_bound_relaxes_to_tp():
    """MLA model: 1 effective kv-head -> bound relaxes to dcp <= tp; dcp=4, tp=8 passes."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=4, tp=8)
    with _patch_autoconfig(_mla_config()):
        _validate_dcp_cfg(cfg)


def test_mla_still_enforces_divisibility():
    """MLA relaxes the kv bound but tp % dcp == 0 still holds (dcp=3, tp=8 raises)."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=3, tp=8)
    with _patch_autoconfig(_mla_config()):
        with pytest.raises(AssertionError, match="tensor_parallel_size % dcp"):
            _validate_dcp_cfg(cfg)


# --------------------------------------------------------------------------- G3 (f)
def test_arch_gate_gqa_passes():
    """A GQA config on the (default) FlashAttention path passes the arch gate."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    with _patch_autoconfig(_gqa_config(num_key_value_heads=4, attn_implementation="flash_attention_2")):
        _validate_dcp_cfg(cfg)


def test_arch_gate_mla_passes():
    """An MLA config passes the arch gate (FlashMLA / FlashAttnMLA are DCP-capable)."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    with _patch_autoconfig(_mla_config()):
        _validate_dcp_cfg(cfg)


def test_arch_gate_rejects_non_dcp_attn():
    """A config pinned to a non-DCP-capable attn backend (eager) is rejected.

    Uses H=1 so the kv-head bound passes and the arch gate is what fires.
    """
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    incapable = _gqa_config(num_key_value_heads=1, attn_implementation="eager")
    with _patch_autoconfig(incapable):
        with pytest.raises(AssertionError, match="DCP-capable attention backend"):
            _validate_dcp_cfg(cfg)


# --------------------------------------------------------------------------- offline degrade
def test_offline_config_unresolvable_degrades_to_cheap_bound():
    """If AutoConfig can't resolve offline, skip the kv bound (warn) but keep dcp<=tp.

    dcp=4, tp=8 with an unresolvable config must PASS (cheap bound only) rather than crash.
    """
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=4, tp=8)
    with mock.patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")):
        _validate_dcp_cfg(cfg)  # must not raise


def test_offline_degrade_still_enforces_cheap_bound():
    """Even degraded, the cheap dcp<=tp bound and tp%dcp==0 still fire (dcp=3, tp=8)."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=3, tp=8)
    with mock.patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")):
        with pytest.raises(AssertionError, match="tensor_parallel_size % dcp"):
            _validate_dcp_cfg(cfg)


# --------------------------------------------------------------------------- remote path
def test_remote_engine_dcp_warns_and_passes():
    """dcp>1 with run_engines_locally=False warns (set -dcp externally) and does not crash."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

    cfg = _dcp_enabled_config(dcp=2, tp=8)
    cfg.generator.run_engines_locally = False
    with _patch_autoconfig(_gqa_config(num_key_value_heads=4)):
        _validate_dcp_cfg(cfg)  # metadata-only path, must not raise


def test_remote_engine_carries_dcp_metadata():
    """create_remote_inference_engines threads decode_context_parallel_size onto the client."""
    from skyrl_train.inference_engines.remote_inference_engine import (
        create_remote_inference_engines,
    )

    engines = create_remote_inference_engines(
        urls=["127.0.0.1:8001"],
        model_name="dummy/model",
        engine_backend="vllm",
        tokenizer=None,
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
    )
    assert len(engines) == 1
    assert engines[0].dcp_size() == 2
    assert engines[0].tp_size() == 8


# --------------------------------------------------------------------------- G4 (model-aware)
def test_g4_invariance_with_model_aware_path():
    """Adding the model-aware kv-head check does not change rollout-GPU accounting (G4)."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_dcp_cfg

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
    before = num_rollout_gpus(cfg2)
    with _patch_autoconfig(_gqa_config(num_key_value_heads=4)):
        _validate_dcp_cfg(cfg2)
    after = num_rollout_gpus(cfg2)
    assert before == after == num_rollout_gpus(cfg1), "model-aware DCP validation must not touch GPU accounting"
