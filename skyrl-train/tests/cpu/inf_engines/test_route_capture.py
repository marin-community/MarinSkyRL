"""Causal alignment of vLLM route rows and generated tokens."""

import base64
import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyrl_train.inference_engines.vllm.route_capture import response_routes
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.config.utils import get_default_config
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.models.megatron_router_replay import (
    SENTINEL_EXPERT_ID,
    LayerReplayHandle,
    MegatronRouterReplay,
)
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts


def test_response_routes_select_prediction_positions() -> None:
    # Three prompt tokens produce the first response token. The next decode
    # routes the first response token to produce the second one.
    captured = np.array([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.uint16)

    response = response_routes(captured, 2)
    np.testing.assert_array_equal(response, [[[5, 6]], [[7, 8]]])
    assert np.issubdtype(response.dtype, np.integer)
    assert not np.shares_memory(response, captured)
    assert response_routes(captured, 0).shape == (0, 1, 2)
    assert response_routes(None, 2) is None


def test_response_routes_reject_incomplete_capture() -> None:
    captured = np.array([[[1, 2]]], dtype=np.uint16)

    with pytest.raises(ValueError, match="1 routed rows for 2 generated tokens"):
        response_routes(captured, 2)


@pytest.mark.parametrize("source", ["generate", "chat"])
@pytest.mark.parametrize("packing", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("mode", ["native", "router_replay"])
@pytest.mark.parametrize(
    ("expert_offset", "num_experts", "expected_dtype"),
    [(0, 32, torch.uint8), (0, 384, torch.int16), (256, 384, torch.int16)],
    ids=["32experts", "384experts_low_ids", "384experts_high_ids"],
)
def test_retry_routes_and_probe_observations_follow_prediction_tokens(
    megatron_wrapper, source, packing, mode, expert_offset, num_experts, expected_dtype
):
    # Each forwarded position and layer has distinct choices, including IDs above 255.
    captured = np.arange(24, dtype=np.uint16).reshape(4, 3, 2) + expert_offset
    chunks = iter([([10, 11, 12], [20], "abort"), ([10, 11, 12, 20], [21], "stop")])

    async def generate(request):
        prompt, tokens, stop = next(chunks)
        assert request["prompt_token_ids"] == [prompt]
        raw = captured[: len(prompt)]
        if source == "generate":
            rows = response_routes(raw, len(tokens))
        else:
            payload = io.BytesIO()
            np.save(payload, raw, allow_pickle=False)
            rows = normalize_routed_experts(base64.b64encode(payload.getvalue()).decode("ascii"), prompt, tokens)
        return {"responses": ["response"], "response_ids": [tokens], "stop_reasons": [stop], "routed_experts": [rows]}

    cfg = get_default_config()
    cfg.generator.enable_http_endpoint = False
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    client = InferenceEngineClient(
        [SimpleNamespace(generate=generate)], SimpleNamespace(decode=lambda *_, **__: "response"), cfg
    )
    output = asyncio.run(client.generate({"prompt_token_ids": [[10, 11, 12]], "sampling_params": {"max_tokens": 2}}))
    assert output["response_ids"] == [[20, 21]]
    response = output["routed_experts"][0]
    np.testing.assert_array_equal(response, captured[2:])
    # Compact batches can carry only the local response rows even when
    # the global padded action window is longer.
    routes = RoutedExpertRows((response,), response_len=3, num_experts=num_experts).materialize()
    assert routes.dtype == expected_dtype
    if mode == "native":
        # The production mismatch probe requests native routing with sentinel rows.
        routes.fill_(SENTINEL_EXPERT_ID)
    sequences = torch.tensor([[0, 0, 0, 10, 11, 12, 20, 21, 0]])
    attention = (sequences != 0).long()
    positions = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 0)
    controller = MegatronRouterReplay(range(3), recompute_enabled=False)
    controller.num_moe_layers_total, controller.topk = 3, 2
    wrapper = megatron_wrapper(packing=packing, controller=controller, num_experts=num_experts)

    def model(tokens, *args, **kwargs):
        tokens = tokens.flatten()
        scores = torch.arange(num_experts, 0, -1, dtype=torch.float32).expand(len(tokens), -1)
        for layer in range(3):
            _, selected = LayerReplayHandle(controller, layer).get_replay_topk(
                scores,
                2,
                default_compute_topk=lambda values, *_args, **_kwargs: torch.topk(values, 2, dim=-1),
            )
            # Routes are attached to the tokens the model forwarded, not the
            # generated tokens whose logprobs are stored in the same rollout rows.
            for token, expected in ((12, captured[2, layer]), (20, captured[3, layer])):
                actual = selected[tokens == token].tolist()
                assert actual == [expected.tolist() if mode == "router_replay" else [0, 1]]
            for token in (10, 11, 21):
                assert selected[tokens == token].tolist() == [[0, 1]]
        return torch.zeros(1)

    wrapper._forward_micro_batch(
        model,
        sequences,
        attention,
        positions,
        rollout_routed_experts=routes,
        probe_row_indices=torch.tensor([0]),
        num_actions=3,
    )
    observations = controller.take_probe_observations()
    collector = ProbeCollector(cfg)
    collector.batch_layout = SimpleNamespace(padded_rows=0)
    collector.probes = [
        SimpleNamespace(
            routed_experts=response.tobytes(),
            routed_experts_shape=list(response.shape),
            route_valid_mask=[[True] * 3] * 2,
            sample_id="sample-0",
        )
    ]
    encoded = collector._route_observations([SimpleNamespace(metadata={"probe_routes": observations})], mode)[0]
    effective = np.frombuffer(encoded.data, dtype=encoded.dtype).reshape(encoded.shape)
    expected = response if mode == "router_replay" else np.broadcast_to([0, 1], response.shape)
    np.testing.assert_array_equal(effective, expected)
