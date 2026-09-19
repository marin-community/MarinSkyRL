"""The trainer-side extraction: a rank's Megatron parameters, sliced as the HF tensors they back."""

from types import SimpleNamespace

import pytest
import torch

from skyrl_train.weight_sync.expert_block.schedule import TrainerRank
from skyrl_train.weight_sync.expert_block.source_views import (
    dense_source_view,
    expert_source_view,
    local_expert_sources,
    local_source_slices,
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
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
    )
    _, expected = reference_hf(parameters)
    assert {(item.entry.projection, item.entry.layer, item.entry.expert) for item in sources} == set(expected)
    for item in sources:
        entry = item.entry
        matrix = expected[entry.projection, entry.layer, entry.expert]
        assert torch.equal(expert_source_view(item, local.sources), matrix.reshape(-1))
        assert entry.nbytes == matrix.numel() * 2


def test_an_expert_outside_the_ranks_block_is_refused():
    local = local_source_slices(conversion_tasks(rank_parameters(experts=(0, 1))), PROVIDER, pp=0)
    with pytest.raises(ValueError, match="not owned by EP rank 1"):
        local_expert_sources(
            local.experts,
            local.sources,
            TrainerRank(rank=1, dp=0, pp=0, ep=1),
            num_experts=NUM_EXPERTS,
            expert_parallel_size=2,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
        )


def task(name, weight, kind):
    return SimpleNamespace(param_weight=weight, global_param_name=name, mapping=kind)


@pytest.mark.parametrize(
    "bad_task,error",
    [
        # Trainer tensor parallelism would make each parameter a shard of its HF tensor.
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
