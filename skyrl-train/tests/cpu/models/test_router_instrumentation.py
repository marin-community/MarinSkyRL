import copy
import sys
import types

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3_5MoeConfig, Qwen3MoeConfig, Qwen3NextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextTopKRouter

from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeRouter

try:
    from skyrl_train.models.layers.moe import TokenChoiceTopKRouter
except ModuleNotFoundError as error:
    if error.name != "torchtitan":
        raise
    torchtitan = types.ModuleType("torchtitan")
    torchtitan_distributed = types.ModuleType("torchtitan.distributed")
    torchtitan_ep = types.ModuleType("torchtitan.distributed.expert_parallel")
    torchtitan_ep.expert_parallel = lambda function: function
    sys.modules["torchtitan"] = torchtitan
    sys.modules["torchtitan.distributed"] = torchtitan_distributed
    sys.modules["torchtitan.distributed.expert_parallel"] = torchtitan_ep
    from skyrl_train.models.layers.moe import TokenChoiceTopKRouter

    del sys.modules["torchtitan.distributed.expert_parallel"]
    del sys.modules["torchtitan.distributed"]
    del sys.modules["torchtitan"]
from skyrl_train.models.router_instrumentation import (
    instrument_moe_routers,
    observe_router_forwards,
)


def _grug_config() -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        shared_expert_intermediate_size=8,
        num_local_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        max_position_embeddings=16,
        sliding_window=4,
    )


def _qwen_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(
        hidden_size=8,
        intermediate_size=16,
        moe_intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=True,
        vocab_size=32,
    )


def test_grouped_qwen_router_observation_matches_native_routing() -> None:
    router = TokenChoiceTopKRouter(dim=4, num_experts=4, top_k=2, route_norm=True)
    assert instrument_moe_routers(router) == 1
    with torch.no_grad():
        router.gate.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0],
                    [0.0, 0.0, 3.0, 0.0],
                    [0.0, 0.0, 0.0, 4.0],
                ]
            )
        )
    hidden = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 3.0, 4.0]])
    observations = []

    with observe_router_forwards(observations.append):
        combine_weights, selected_experts, _ = router(hidden)

    logits = F.linear(hidden, router.gate.weight)
    expected_probabilities = logits.softmax(dim=-1)
    expected_weights, expected_experts = expected_probabilities.topk(2, dim=-1)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)

    assert len(observations) == 1
    observation = observations[0]
    torch.testing.assert_close(observation.router_inputs, hidden)
    torch.testing.assert_close(observation.selection_logits, logits)
    torch.testing.assert_close(observation.selection_log_probs, logits.log_softmax(dim=-1))
    torch.testing.assert_close(observation.natural_selected_experts, expected_experts)
    torch.testing.assert_close(observation.selected_experts, selected_experts)
    torch.testing.assert_close(observation.combine_weights, combine_weights)
    torch.testing.assert_close(selected_experts, expected_experts)
    torch.testing.assert_close(combine_weights, expected_weights)


def test_grouped_qwen_router_observation_distinguishes_replayed_route() -> None:
    router = TokenChoiceTopKRouter(dim=4, num_experts=4, top_k=2, route_norm=True)
    hidden = torch.randn(3, 4)
    replayed_experts = torch.tensor([[2, 3], [1, 3], [0, 2]])
    observations = []

    with observe_router_forwards(observations.append):
        combine_weights, selected_experts, _ = router(hidden, routed_experts=replayed_experts)

    scores = router.gate(hidden).float().softmax(dim=-1)
    natural_experts = scores.topk(2, dim=-1).indices
    replayed_weights = scores.gather(-1, replayed_experts)
    replayed_weights /= replayed_weights.sum(dim=-1, keepdim=True)

    assert len(observations) == 1
    observation = observations[0]
    torch.testing.assert_close(observation.natural_selected_experts, natural_experts)
    torch.testing.assert_close(observation.selected_experts, replayed_experts)
    torch.testing.assert_close(observation.combine_weights, replayed_weights)
    torch.testing.assert_close(selected_experts, replayed_experts)
    torch.testing.assert_close(combine_weights, replayed_weights)


def test_grug_router_observation_uses_biased_selection_and_native_weights() -> None:
    router = GrugMoeRouter(_grug_config())
    assert instrument_moe_routers(router) == 1
    with torch.no_grad():
        router.weight.copy_(torch.eye(4, 8))
        router.bias.copy_(torch.tensor([0.0, 0.5, -0.25, 0.75]))
    hidden = torch.tensor([[4.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
    observations = []

    with observe_router_forwards(observations.append):
        raw_logits, selected_experts, combine_weights = router(hidden)

    selection_logits = raw_logits + router.bias
    selected_raw_logits = raw_logits.gather(-1, selected_experts)
    expected_weights = selected_raw_logits.sigmoid()
    expected_weights *= 2.5 / expected_weights.sum(dim=-1, keepdim=True)

    assert len(observations) == 1
    observation = observations[0]
    torch.testing.assert_close(observation.router_inputs, hidden)
    torch.testing.assert_close(observation.selection_logits, selection_logits)
    torch.testing.assert_close(observation.selection_log_probs, selection_logits.log_softmax(dim=-1))
    torch.testing.assert_close(observation.natural_selected_experts, selected_experts)
    torch.testing.assert_close(observation.selected_experts, selected_experts)
    torch.testing.assert_close(observation.combine_weights, combine_weights)
    torch.testing.assert_close(combine_weights, expected_weights)


def test_qwen_eager_instrumentation_preserves_output_and_router_gradients() -> None:
    torch.manual_seed(17)
    reference = Qwen3MoeSparseMoeBlock(_qwen_config())
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.normal_(mean=0.0, std=0.1)
    instrumented = copy.deepcopy(reference)
    hidden = torch.randn(2, 3, 8)
    expected = reference(hidden)
    observations = []

    assert instrument_moe_routers(instrumented) == 1
    assert instrument_moe_routers(instrumented) == 1
    with observe_router_forwards(observations.append):
        actual = instrumented(hidden)

    router_logits, routing_weights, selected_experts = instrumented.gate(hidden.reshape(-1, hidden.shape[-1]))
    assert len(observations) == 1
    observation = observations[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(observation.router_inputs, hidden.reshape(-1, hidden.shape[-1]))
    torch.testing.assert_close(observation.selection_logits, router_logits)
    torch.testing.assert_close(observation.selection_log_probs, router_logits.float().log_softmax(dim=-1))
    torch.testing.assert_close(observation.natural_selected_experts, selected_experts)
    torch.testing.assert_close(observation.selected_experts, selected_experts)
    torch.testing.assert_close(observation.combine_weights, routing_weights)

    actual.sum().backward()
    assert torch.count_nonzero(instrumented.gate.weight.grad) > 0


@pytest.mark.parametrize(
    ("config_type", "router_type"),
    [
        (Qwen3NextConfig, Qwen3NextTopKRouter),
        (Qwen3_5MoeConfig, Qwen3_5MoeTopKRouter),
    ],
)
def test_other_qwen_router_variants_preserve_their_native_return_tuple(config_type, router_type) -> None:
    config = config_type(hidden_size=8, num_experts=4, num_experts_per_tok=2, norm_topk_prob=True)
    router = router_type(config)
    with torch.no_grad():
        router.weight.normal_(mean=0.0, std=0.1)
    hidden = torch.randn(2, 3, 8)
    expected = router(hidden)
    observations = []

    assert instrument_moe_routers(router) == 1
    with observe_router_forwards(observations.append):
        actual = router(hidden)

    assert len(observations) == 1
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
    observation = observations[0]
    torch.testing.assert_close(observation.router_inputs, hidden.reshape(-1, hidden.shape[-1]))
    torch.testing.assert_close(observation.selection_logits, actual[0])
    torch.testing.assert_close(observation.natural_selected_experts, actual[2])
    torch.testing.assert_close(observation.combine_weights, actual[1])
