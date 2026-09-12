from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.frozen_source_views import local_source_slices, source_view


def task(kind, name, hf, source):
    mapping = type(kind, (), {})()
    mapping.hf_param = hf
    return SimpleNamespace(mapping=mapping, global_param_name=name, param_weight=source)


def config():
    return SimpleNamespace(
        tensor_model_parallel_size=1,
        num_moe_experts=4,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=2,
        hidden_size=3,
    )


def test_qkv_and_gated_views_match_full_exports_without_new_storage():
    qkv = torch.arange(48, dtype=torch.bfloat16).reshape(16, 3)
    gated = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
    tasks = [
        task("QKVMapping", "qkv.weight", {"q": "q", "k": "k", "v": "v"}, qkv),
        task("GatedMLPMapping", "gated.weight", {"gate": "gate", "up": "up"}, gated),
    ]
    slices, sources = local_source_slices(tasks, config())
    group = qkv.reshape(2, 4, 2, 3)
    expected = {
        "q": group[:, :2].reshape(-1),
        "k": group[:, 2].reshape(-1),
        "v": group[:, 3].reshape(-1),
        "gate": gated[:4].reshape(-1),
        "up": gated[4:].reshape(-1),
    }
    for name, reference in expected.items():
        ordered = sorted((item for item in slices if item.hf_name == name), key=lambda item: item.hf_offset)
        values = [source_view(item, sources) for item in ordered]
        assert torch.equal(torch.cat(values).view(torch.uint8), reference.view(torch.uint8))
        assert all(
            value.untyped_storage().data_ptr() == sources[item.source_key].untyped_storage().data_ptr()
            for item, value in zip(ordered, values, strict=True)
        )


def test_global_expert_offsets_and_fp32_bias_preserve_original_bits():
    expert = torch.arange(24, dtype=torch.bfloat16).reshape(8, 3)
    bias = torch.tensor([-0.0, float("nan")], dtype=torch.float32)
    tasks = [
        task(
            "GrugStackedGatedExpertMapping",
            "decoder.layers.9.mlp.experts.linear_fc1.weight3",
            {"gate": "gate", "up": "up"},
            expert,
        ),
        task("ReplicatedMapping", "router.expert_bias", "bias", bias),
    ]
    slices, sources = local_source_slices(tasks, config())
    assert [(item.hf_name, item.hf_offset, item.source_offset) for item in slices] == [
        ("gate", 36, 0),
        ("up", 36, 12),
        ("bias", 0, 0),
    ]
    assert torch.equal(source_view(slices[-1], sources).view(torch.uint8), bias.view(torch.uint8))


@pytest.mark.parametrize("failure", ["tp", "grouped", "expert", "mapping", "qkv_geometry"])
def test_unqualified_native_source_layout_rejects(failure):
    settings = config()
    source = torch.ones(8, 3, dtype=torch.bfloat16)
    value = task("GrugStackedGatedExpertMapping", "experts.linear_fc1.weight0", {"gate": "gate", "up": "up"}, source)
    if failure == "tp":
        settings.tensor_model_parallel_size = 2
    elif failure == "grouped":
        value.param_weight = source.reshape(2, 4, 3)
    elif failure == "expert":
        value.global_param_name = "experts.linear_fc1.weight4"
    elif failure == "mapping":
        value.mapping = type("UnverifiedMapping", (), {})()
    else:
        value = task("QKVMapping", "qkv.weight", {"q": "q", "k": "k", "v": "v"}, source)
    with pytest.raises(ValueError):
        local_source_slices([value], settings)
