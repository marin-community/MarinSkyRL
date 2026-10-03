"""Model-aware validation of vLLM Decode Context Parallel (DCP) rollout settings.

When DCP is enabled, `_validate_dcp_cfg` resolves the policy's HF config to bound
dcp by tp // num_kv_heads (relaxed to tp for MLA), gates on a DCP-capable attention
backend, and degrades to the cheap dcp <= tp bound when the config is unresolvable.
Every case patches `AutoConfig.from_pretrained` so no test reaches the HF hub.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from skyrl_train.config.utils import get_default_config
from skyrl_train.inference_engines.remote_inference_engine import create_remote_inference_engines
from skyrl_train.utils.utils import _validate_dcp_cfg

TP = 8


def _gqa(num_key_value_heads, attn_implementation=None):
    cfg = SimpleNamespace(
        num_key_value_heads=num_key_value_heads,
        num_attention_heads=32,
        architectures=["Qwen3ForCausalLM"],
    )
    if attn_implementation is not None:
        cfg._attn_implementation = attn_implementation
    return cfg


MLA = SimpleNamespace(
    kv_lora_rank=512, q_lora_rank=1536, num_attention_heads=128, architectures=["DeepseekV3ForCausalLM"]
)
MHA_32_HEADS = SimpleNamespace(num_attention_heads=32, architectures=["LlamaForCausalLM"])
OFFLINE = OSError("offline")


@pytest.mark.parametrize(
    ("dcp", "model_config", "match"),
    [
        pytest.param(2, _gqa(4, attn_implementation="flash_attention_2"), None, id="gqa-within-bound"),
        pytest.param(4, _gqa(2), None, id="gqa-at-bound"),
        pytest.param(8, _gqa(4), "kv-head bound", id="gqa-above-bound"),
        pytest.param(2, _gqa(8), "kv-head bound", id="gqa-no-headroom"),
        pytest.param(2, MHA_32_HEADS, "kv-head bound", id="mha-uses-attention-heads"),
        pytest.param(4, MLA, None, id="mla-relaxes-to-tp"),
        pytest.param(2, _gqa(1, attn_implementation="eager"), "DCP-capable attention backend", id="eager-attention"),
        pytest.param(4, OFFLINE, None, id="unresolvable-config-degrades-to-cheap-bound"),
    ],
)
def test_dcp_model_aware_bounds(dcp, model_config, match):
    cfg = get_default_config()
    cfg.generator.backend = "vllm"
    cfg.generator.inference_engine_tensor_parallel_size = TP
    cfg.generator.inference_engine_decode_context_parallel_size = dcp
    resolved = {"side_effect": model_config} if model_config is OFFLINE else {"return_value": model_config}

    with mock.patch("transformers.AutoConfig.from_pretrained", **resolved):
        if match is None:
            _validate_dcp_cfg(cfg)
        else:
            with pytest.raises(AssertionError, match=match):
                _validate_dcp_cfg(cfg)


def test_remote_engine_carries_dcp_metadata():
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
