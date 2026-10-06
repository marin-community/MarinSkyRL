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


@pytest.mark.parametrize("expert_schema", ["stacked", "split"])
@pytest.mark.parametrize("expert_width", [HIDDEN, HIDDEN - 1])
def test_source_slices_reconstruct_dense_and_owned_expert_matrices(expert_schema, expert_width):
    parameters = megatron_parameters(
        LAYERS, (2, 3), last_stage=True, model_names=MODEL_NAMES, expert_hidden_size=expert_width
    )
    sconv_name = "model.layers.0.self_attn.sconv_k.weight"
    sconv = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    tasks = conversion_tasks(parameters, expert_schema=expert_schema)
    tasks.append(
        task("decoder.layers.0.self_attention.sconv_k.weight", sconv, mapping("RowParallelMapping", sconv_name))
    )
    dense, experts = reference_hf({name: weight.clone() for name, weight in parameters.items()})
    dense[sconv_name] = sconv.clone()
    local = local_source_slices(tasks, PROVIDER, pp=0)
    assembled = {name: torch.zeros_like(value) for name, value in dense.items()}
    written = {name: 0 for name in dense}
    for item in local.dense:
        run = assembled[item.hf_name].view(-1).narrow(0, item.hf_offset, item.numel)
        run.copy_(dense_source_view(item, local.sources))
        written[item.hf_name] += item.numel
    assert written == {name: value.numel() for name, value in dense.items()}
    for name, value in dense.items():
        assert torch.equal(assembled[name], value), name
    # EP rank 1 of 2 owns experts 2 and 3.
    sources = local_expert_sources(
        local.experts,
        local.sources,
        TrainerRank(rank=1, dp=0, pp=0, ep=1),
        num_experts=NUM_EXPERTS,
        expert_parallel_size=2,
        expert_hidden_size=expert_width,
        intermediate_size=INTERMEDIATE,
    )
    assert {(item.entry.projection, item.entry.layer, item.entry.expert) for item in sources} == set(experts)
    for item in sources:
        entry = item.entry
        matrix = experts[entry.projection, entry.layer, entry.expert]
        assert torch.equal(expert_source_view(item, local.sources), matrix.reshape(-1)), entry.name
        assert entry.nbytes == matrix.numel() * 2


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
