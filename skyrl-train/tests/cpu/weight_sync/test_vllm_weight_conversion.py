import pytest
import torch
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

from skyrl_train.weight_sync.weight_extractor_utils import yield_module_grouped_chunks
from skyrl_train.weight_sync.vllm_weight_conversion import expected_vllm_parameter_names, load_weights_into_vllm


class RecordingVLLMModel:
    def __init__(self, loaded_parameters: set[str]):
        self.loaded_parameters = loaded_parameters
        self.weights: dict[str, torch.Tensor] = {}

    def load_weights(self, weights):
        self.weights.update(weights)
        return self.loaded_parameters


class PerSourceRecordingVLLMModel:
    def __init__(self, ignored_sources: set[str] | None = None):
        self.ignored_sources = ignored_sources or set()
        self.calls: list[set[str]] = []

    def load_weights(self, weights):
        names = {name for name, _ in weights}
        self.calls.append(names)
        loaded = set()
        for name in names - self.ignored_sources:
            loaded.update(_packed_reported_names(name))
        return loaded


def _packed_reported_names(name: str) -> set[str]:
    for source, target in (
        ("self_attn.q_proj", "self_attn.qkv_proj"),
        ("self_attn.k_proj", "self_attn.qkv_proj"),
        ("self_attn.v_proj", "self_attn.qkv_proj"),
        ("mlp.gate_proj", "mlp.gate_up_proj"),
        ("mlp.up_proj", "mlp.gate_up_proj"),
    ):
        if source in name:
            return {name.replace(source, target)}
    return {name}


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


def test_load_weights_into_vllm_rejects_silently_skipped_ordinary_parameter():
    model = RecordingVLLMModel({"model.embed_tokens.weight"})

    with pytest.raises(RuntimeError, match=r"model\.norm\.weight"):
        load_weights_into_vllm(
            model,
            [
                ("model.embed_tokens.weight", torch.zeros(4, 4)),
                ("model.norm.weight", torch.zeros(4)),
            ],
        )


def test_load_weights_into_vllm_accepts_qwen_packed_parameter_names():
    model = PerSourceRecordingVLLMModel()
    source_names = {
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
    }

    loaded = load_weights_into_vllm(
        model,
        [(name, torch.zeros(4, 4)) for name in sorted(source_names)],
    )

    assert loaded == source_names
    assert len(model.calls) == len(source_names)
    assert all(len(call) == 1 for call in model.calls)


def test_load_weights_into_vllm_rejects_one_skipped_qwen_packed_source():
    skipped = "model.layers.0.self_attn.k_proj.weight"
    model = PerSourceRecordingVLLMModel({skipped})

    with pytest.raises(RuntimeError, match=r"model\.layers\.0\.self_attn\.k_proj\.weight"):
        load_weights_into_vllm(
            model,
            [
                ("model.layers.0.self_attn.q_proj.weight", torch.zeros(4, 4)),
                (skipped, torch.zeros(4, 4)),
                ("model.layers.0.self_attn.v_proj.weight", torch.zeros(4, 4)),
            ],
        )


def test_expected_parameter_names_cover_grug_stacked_experts():
    assert expected_vllm_parameter_names(
        [
            "model.layers.0.mlp.experts.gate_proj.weight",
            "model.layers.0.mlp.experts.up_proj.weight",
            "model.layers.0.mlp.experts.down_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
    ) == {
        "model.layers.0.mlp.experts.routed_experts.w13_weight",
        "model.layers.0.mlp.experts.routed_experts.w2_weight",
        "model.layers.0.self_attn.q_proj.weight",
    }


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
