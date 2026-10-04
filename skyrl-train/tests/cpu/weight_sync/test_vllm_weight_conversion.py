import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks
from skyrl_train.weight_sync.vllm_weight_conversion import load_weights_into_vllm, validate_dummy_weight_coverage


class RecordingVLLMModel:
    def __init__(self, loaded_parameters: set[str]):
        self.loaded_parameters = loaded_parameters
        self.weights: dict[str, torch.Tensor] = {}

    def load_weights(self, weights):
        self.weights = dict(weights)
        return self.loaded_parameters


def test_load_weights_into_vllm_expands_transformers_fused_moe_weights():
    gate_up = torch.arange(2 * 6 * 4, dtype=torch.float32).reshape(2, 6, 4)
    down = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    router = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4)
    policy = Qwen3MoeForCausalLM(
        Qwen3MoeConfig(
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            moe_intermediate_size=3,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            num_experts=2,
            num_experts_per_tok=1,
        )
    )
    policy_state = policy.state_dict()
    policy_state["model.layers.0.mlp.experts.gate_up_proj"].copy_(gate_up)
    policy_state["model.layers.0.mlp.experts.down_proj"].copy_(down)
    policy_state["model.layers.0.mlp.gate.weight"].copy_(router)
    policy_weights = {name: tensor for name, tensor in policy_state.items() if ".mlp." in name}
    chunks = yield_module_grouped_chunks(
        policy_weights,
        dtype=torch.float32,
        gather_tensor_fn=lambda tensor: tensor,
        get_shape_fn=lambda _name, _parameter, tensor: list(tensor.shape),
        batch_size_threshold_gb=1.0,
    )
    transferred_weights = [
        (name, tensor) for chunk in chunks for name, tensor in zip(chunk.names, chunk.tensors, strict=True)
    ]
    model = RecordingVLLMModel(
        {
            "model.layers.0.mlp.experts.w13_weight",
            "model.layers.0.mlp.experts.w2_weight",
            "model.layers.0.mlp.gate.weight",
        }
    )

    loaded = load_weights_into_vllm(
        model,
        transferred_weights,
    )

    assert loaded == model.loaded_parameters
    assert set(model.weights) == {
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        "model.layers.0.mlp.experts.0.up_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.0.mlp.experts.1.gate_proj.weight",
        "model.layers.0.mlp.experts.1.up_proj.weight",
        "model.layers.0.mlp.experts.1.down_proj.weight",
        "model.layers.0.mlp.gate.weight",
    }
    torch.testing.assert_close(model.weights["model.layers.0.mlp.experts.0.gate_proj.weight"], gate_up[0, :3])
    torch.testing.assert_close(model.weights["model.layers.0.mlp.experts.0.up_proj.weight"], gate_up[0, 3:])
    torch.testing.assert_close(model.weights["model.layers.0.mlp.experts.1.gate_proj.weight"], gate_up[1, :3])
    torch.testing.assert_close(model.weights["model.layers.0.mlp.experts.1.up_proj.weight"], gate_up[1, 3:])
    torch.testing.assert_close(model.weights["model.layers.0.mlp.experts.0.down_proj.weight"], down[0])
    torch.testing.assert_close(model.weights["model.layers.0.mlp.experts.1.down_proj.weight"], down[1])
    torch.testing.assert_close(model.weights["model.layers.0.mlp.gate.weight"], router)


def test_load_weights_into_vllm_rejects_silently_skipped_fused_experts():
    model = RecordingVLLMModel({"model.layers.0.mlp.experts.w13_weight"})

    with pytest.raises(RuntimeError, match=r"model\.layers\.0\.mlp\.experts\.w2_weight"):
        load_weights_into_vllm(
            model,
            [
                ("model.layers.0.mlp.experts.gate_up_proj", torch.zeros(2, 6, 4)),
                ("model.layers.0.mlp.experts.down_proj", torch.zeros(2, 4, 3)),
            ],
        )


@pytest.mark.parametrize(
    ("name", "tensor"),
    [
        ("model.layers.0.mlp.experts.gate_up_proj", torch.zeros(2, 5, 4)),
        ("model.layers.0.mlp.experts.down_proj", torch.zeros(2, 4)),
    ],
)
def test_load_weights_into_vllm_rejects_invalid_fused_expert_shapes(name, tensor):
    model = RecordingVLLMModel(set())

    with pytest.raises(ValueError, match="fused MoE weight"):
        load_weights_into_vllm(model, [(name, tensor)])


@pytest.mark.parametrize(
    ("loaded", "total", "padding", "bias_padding", "non_persistent", "complete", "fails"),
    [
        pytest.param(12, 12, 0, 0, 0, False, False, id="complete-weight"),
        pytest.param(0, 12, 0, 0, 0, False, True, id="unsent-layer"),
        pytest.param(8, 12, 0, 0, 0, False, True, id="missing-stacked-part"),
        pytest.param(12, 15, 0, 0, 0, False, True, id="unsent-bias"),
        pytest.param(0, None, 0, 0, 0, True, False, id="already-processed"),
        pytest.param(0, 32, 0, 0, 32, False, False, id="non-persistent-only"),
        pytest.param(0, 44, 0, 0, 32, False, True, id="non-persistent-with-unsent-parameter"),
        pytest.param(12, 16, 4, 0, 0, False, False, id="vocabulary-padding"),
        pytest.param(8, 16, 4, 0, 0, False, True, id="padding-does-not-cover-unsent-weight"),
        pytest.param(252, 256, 3, 1, 0, False, False, id="padded-vocabulary-bias"),
        pytest.param(189, 256, 3, 1, 0, False, True, id="unsent-padded-vocabulary-bias"),
    ],
)
def test_dummy_weights_require_every_loadable_layer_element(
    loaded, total, padding, bias_padding, non_persistent, complete, fails
):
    layers = {
        "model.layer": {
            "can_load": not complete,
            "load_numel": loaded,
            "load_numel_total": total,
            "tensors": {},
            "vocab_padding_numel": padding,
            "vocab_bias_padding_numel": bias_padding,
            "non_persistent_numel": non_persistent,
        }
    }
    if fails:
        with pytest.raises(RuntimeError, match="model.layer"):
            validate_dummy_weight_coverage(layers, set(), set())
    else:
        validate_dummy_weight_coverage(layers, set(), set())


@pytest.mark.parametrize(
    ("bias_numel", "loaded_bias", "fails"),
    [(0, 0, False), (4, 4, False), (4, 0, True)],
    ids=["tied-head", "tied-head-bias", "unsent-tied-head-bias"],
)
@pytest.mark.parametrize("embedding_processed", [True, False])
def test_dummy_tied_head_requires_its_bias_after_shared_weight_and_padding(
    bias_numel, loaded_bias, fails, embedding_processed
):
    weight = torch.ones(4, 3)
    identity = (weight.data_ptr(), tuple(weight.shape), tuple(weight.stride()), str(weight.dtype), str(weight.device))
    layers = {
        "model.embed_tokens": {
            "can_load": not embedding_processed,
            "load_numel": 6,
            "load_numel_total": None if embedding_processed else weight.numel(),
            "tensors": {"weight": (identity, weight.numel())},
            "vocab_padding_numel": 6,
        },
        "lm_head": {
            "can_load": True,
            "load_numel": loaded_bias,
            "load_numel_total": weight.numel() + bias_numel,
            "tensors": {"weight": (identity, weight.numel())},
            "vocab_padding_numel": 6,
        },
    }
    if fails:
        with pytest.raises(RuntimeError, match="lm_head"):
            validate_dummy_weight_coverage(layers, set(), set())
    else:
        validate_dummy_weight_coverage(layers, set(), set())


@pytest.mark.parametrize("loaded", [True, False])
def test_dummy_float_weights_excluded_from_layer_counts_still_require_a_load(loaded):
    name = "model.layer.e_score_correction_bias"
    if loaded:
        validate_dummy_weight_coverage({}, {name}, {name})
    else:
        with pytest.raises(RuntimeError, match="e_score_correction_bias"):
            validate_dummy_weight_coverage({}, set(), {name})


@pytest.mark.parametrize(
    ("total", "fails"), [(4, False), (16, True)], ids=["generated-scales", "unsent-weight-with-scales"]
)
def test_dummy_generated_attention_scales_do_not_exempt_checkpoint_weights(total, fails):
    layers = {
        "model.attn": {
            "can_load": True,
            "load_numel": 0,
            "load_numel_total": total,
            "tensors": {},
            "vocab_padding_numel": 0,
            "generated_numel": 4,
        }
    }
    if fails:
        with pytest.raises(RuntimeError, match="model.attn"):
            validate_dummy_weight_coverage(layers, set(), set())
    else:
        validate_dummy_weight_coverage(layers, set(), set())


@pytest.mark.parametrize("missing", [0, 393216, 1536], ids=["complete", "missing-gate", "missing-bias"])
def test_dummy_moe_backend_padding_preserves_checkpoint_coverage(missing):
    layers = {
        "model.experts": {
            "can_load": True,
            "load_numel_total": 3150848,
            "load_numel": 2363392 - missing,
            "tensors": {},
            "vocab_padding_numel": 0,
            "tensor_padding_numel": {"w13_weight": 524288, "w2_weight": 262144, "w2_bias": 1024},
        }
    }
    if missing:
        with pytest.raises(RuntimeError, match="model.experts"):
            validate_dummy_weight_coverage(layers, set(), set())
    else:
        validate_dummy_weight_coverage(layers, set(), set())
