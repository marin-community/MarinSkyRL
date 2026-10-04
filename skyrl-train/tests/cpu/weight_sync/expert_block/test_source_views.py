"""``local_source_slices`` turns a rank's Megatron parameters into slices of HF tensors."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.expert_block.schedule import TrainerRank
from skyrl_train.weight_sync.expert_block.megatron_source import local_source_slices
from skyrl_train.weight_sync.expert_block.source_views import (
    dense_source_view,
    expert_source_view,
    local_expert_sources,
)
from tests.cpu.weight_sync.expert_block.megatron_layout import (
    HIDDEN,
    INTERMEDIATE,
    NUM_EXPERTS,
    PROVIDER,
    conversion_tasks,
    mapping,
    megatron_parameters,
    megatron_shapes,
    reference_hf,
)

LAYERS = (0, 1)
MODEL_NAMES = sorted(megatron_shapes(LAYERS, range(NUM_EXPERTS), last_stage=True))


def rank_parameters(experts=range(NUM_EXPERTS)):
    return megatron_parameters(LAYERS, experts, last_stage=True, model_names=MODEL_NAMES)


def test_dense_slices_assemble_every_hf_tensor_from_the_interleaved_and_fused_parameters():
    parameters = rank_parameters()
    local = local_source_slices(conversion_tasks(parameters), PROVIDER, pp=0)
    expected, _ = reference_hf(parameters)
    assembled = {name: torch.zeros_like(value) for name, value in expected.items()}
    written = {name: 0 for name in expected}
    for item in local.dense:
        run = assembled[item.hf_name].view(-1).narrow(0, item.hf_offset, item.numel)
        run.copy_(dense_source_view(item, local.sources))
        written[item.hf_name] += item.numel
    assert written == {name: value.numel() for name, value in expected.items()}
    for name, value in expected.items():
        assert torch.equal(assembled[name], value), name


def test_hero_sconv_row_parallel_weight_is_sent_whole_at_tp_one():
    name = "model.layers.0.self_attn.sconv_k.weight"
    weight = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    local = local_source_slices(
        [task("decoder.layers.0.self_attention.sconv_k.weight", weight, mapping("RowParallelMapping", name))],
        PROVIDER,
        pp=0,
    )
    assert len(local.dense) == 1
    assert local.dense[0].hf_name == name
    assert torch.equal(dense_source_view(local.dense[0], local.sources), weight.flatten())


def test_expert_sources_are_the_whole_gate_up_and_down_matrices_of_the_ranks_own_block():
    # EP rank 1 of 2 owns experts 2 and 3.
    parameters = rank_parameters(experts=(2, 3))
    local = local_source_slices(conversion_tasks(parameters), PROVIDER, pp=0)
    sources = local_expert_sources(
        local.experts,
        local.sources,
        TrainerRank(rank=1, dp=0, pp=0, ep=1),
        num_experts=NUM_EXPERTS,
        expert_parallel_size=2,
        expert_hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
    )
    _, expected = reference_hf(parameters)
    assert {(item.entry.projection, item.entry.layer, item.entry.expert) for item in sources} == set(expected)
    for item in sources:
        entry = item.entry
        matrix = expected[entry.projection, entry.layer, entry.expert]
        assert torch.equal(expert_source_view(item, local.sources), matrix.reshape(-1))
        assert entry.nbytes == matrix.numel() * 2


def test_split_schema_expert_mappings_use_the_expert_schedule():
    # Split-schema mappings expose each expert as separate HF tensors.
    prefix = "model.layers.0.mlp.experts.2"
    latent_size = HIDDEN - 1
    fc1 = torch.arange(2 * INTERMEDIATE * latent_size, dtype=torch.bfloat16).reshape(2 * INTERMEDIATE, latent_size)
    fc2 = torch.arange(latent_size * INTERMEDIATE, dtype=torch.bfloat16).reshape(latent_size, INTERMEDIATE)
    local = local_source_slices(
        [
            task(
                "decoder.layers.0.mlp.experts.linear_fc1.weight2",
                fc1,
                mapping("GatedMLPMapping", {part: f"{prefix}.{part}_proj.weight" for part in ("gate", "up")}),
            ),
            task(
                "decoder.layers.0.mlp.experts.linear_fc2.weight2",
                fc2,
                mapping("AutoMapping", f"{prefix}.down_proj.weight"),
            ),
        ],
        PROVIDER,
        pp=0,
    )
    assert local.dense == []
    sources = local_expert_sources(
        local.experts,
        local.sources,
        TrainerRank(rank=1, dp=0, pp=0, ep=1),
        num_experts=NUM_EXPERTS,
        expert_parallel_size=2,
        expert_hidden_size=latent_size,
        intermediate_size=INTERMEDIATE,
    )
    assert {(item.entry.layer, item.entry.expert, item.entry.projection) for item in sources} == {
        (0, 2, "fc1"),
        (0, 2, "fc2"),
    }
    assert torch.equal(expert_source_view(sources[0], local.sources), fc1.flatten())
    assert torch.equal(expert_source_view(sources[1], local.sources), fc2.flatten())


def test_split_schema_rejects_a_mismatched_expert_id():
    bad = task(
        "decoder.layers.0.mlp.experts.linear_fc2.weight2",
        torch.zeros(HIDDEN, INTERMEDIATE, dtype=torch.bfloat16),
        mapping("AutoMapping", "model.layers.0.mlp.experts.3.down_proj.weight"),
    )
    with pytest.raises(ValueError, match="disagrees with its HF tensors"):
        local_source_slices([bad], PROVIDER, pp=0)


def test_an_expert_outside_the_ranks_block_is_refused():
    local = local_source_slices(conversion_tasks(rank_parameters(experts=(0, 1))), PROVIDER, pp=0)
    with pytest.raises(ValueError, match="not owned by EP rank 1"):
        local_expert_sources(
            local.experts,
            local.sources,
            TrainerRank(rank=1, dp=0, pp=0, ep=1),
            num_experts=NUM_EXPERTS,
            expert_parallel_size=2,
            expert_hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
        )


def task(name, weight, kind):
    return SimpleNamespace(param_weight=weight, global_param_name=name, mapping=kind)


@pytest.mark.parametrize(
    "bad_task,error",
    [
        # With trainer TP a parameter is only a shard of its HF tensor.
        (
            task(
                "output_layer.weight", torch.zeros(5, 3, dtype=torch.bfloat16), mapping("AutoMapping", "w", tp_size=2)
            ),
            "requires trainer TP=1",
        ),
        (
            task("norm.weight", torch.zeros(3, dtype=torch.float16), mapping("ReplicatedMapping", "w")),
            "contiguous and BF16 or FP32",
        ),
        (
            task("proj.weight", torch.zeros(3, 4, dtype=torch.bfloat16).t(), mapping("AutoMapping", "w")),
            "contiguous and BF16 or FP32",
        ),
        (
            task(
                "fc1.weight",
                torch.zeros(3, 3, dtype=torch.bfloat16),
                mapping("GatedMLPMapping", {"gate": "g", "up": "u"}),
            ),
            "not a complete .gate;up. matrix",
        ),
        (
            task(
                "qkv.weight",
                torch.zeros(7, 3, dtype=torch.bfloat16),
                mapping("QKVMapping", {"q": "q", "k": "k", "v": "v"}),
            ),
            "differs from the configured interleaved layout",
        ),
        (
            task("conv.weight", torch.zeros(3, dtype=torch.bfloat16), mapping("ConvMapping", "w")),
            "Unsupported weight mapping",
        ),
    ],
)
def test_parameters_the_transport_cannot_slice_are_refused(bad_task, error):
    with pytest.raises(ValueError, match=error):
        local_source_slices([bad_task], PROVIDER, pp=0)
