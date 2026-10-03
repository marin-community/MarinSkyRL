"""Causal alignment of vLLM route rows and generated tokens."""

import base64
import asyncio
from contextlib import nullcontext
import io
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyrl_train.inference_engines.vllm.route_capture import response_routes
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.mismatch_probe.collect import ProbeCollector
from skyrl_train.models.megatron_router_replay import LayerReplayHandle, MegatronRouterReplay, dense_replay_targets
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts
from tests.cpu.test_megatron_router_replay_plumbing import _make_wrapper
from tests.cpu.inf_engines.test_inference_engine_client import _make_min_cfg


def test_response_routes_select_prediction_positions() -> None:
    # Three prompt tokens produce the first response token. The next decode
    # routes the first response token to produce the second one.
    captured = np.array([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.uint16)

    response = response_routes(captured, 2)
    np.testing.assert_array_equal(response, [[[5, 6]], [[7, 8]]])
    assert response.dtype == captured.dtype
    assert not np.shares_memory(response, captured)
    assert response_routes(captured, 0).shape == (0, 1, 2)
    assert response_routes(None, 2) is None


def test_response_routes_reject_incomplete_capture() -> None:
    captured = np.array([[[1, 2]]], dtype=np.uint16)

    with pytest.raises(ValueError, match="1 routed rows for 2 generated tokens"):
        response_routes(captured, 2)


@pytest.mark.parametrize("source", ["generate", "chat"])
def test_captured_routes_replay_at_the_positions_that_predicted_responses(source: str) -> None:
    # The model sees prompt tokens 10, 11, 12 before generating 20 and 21.
    captured = np.array([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.uint16)
    if source == "generate":
        response = response_routes(captured, 2)
    else:
        payload = io.BytesIO()
        np.save(payload, captured, allow_pickle=False)
        response = normalize_routed_experts(
            base64.b64encode(payload.getvalue()).decode("ascii"), [10, 11, 12], [20, 21]
        )
    routes = torch.tensor(np.asarray(response)).unsqueeze(0)
    full, mask = dense_replay_targets(routes, batch_size=1, seq_len=5, num_actions=2)

    assert full[0, 2].tolist() == [[5, 6]]  # prompt token 12 predicts response token 20
    assert full[0, 3].tolist() == [[7, 8]]  # response token 20 predicts response token 21
    assert mask[0].tolist() == [False, False, True, True, False]


@pytest.mark.parametrize("packing", [False, True], ids=["unpacked", "packed"])
@pytest.mark.parametrize("mode", ["native", "router_replay"])
@pytest.mark.parametrize("expert_offset", [0, 256], ids=["uint8-routes", "hero-uint16-routes"])
def test_client_retry_compact_routes_and_probe_observations_follow_prediction_tokens(
    monkeypatch, packing, mode, expert_offset
):
    captured = np.array(
        [
            [[1, 2], [1, 2], [1, 2]],
            [[3, 4], [3, 4], [3, 4]],
            [[4, 5], [0, 1], [6, 7]],
            [[5, 6], [1, 2], [7, 0]],
        ],
        dtype=np.uint16,
    )
    captured += expert_offset
    chunks = iter([([10, 11, 12], [20], captured[:3], "abort"), ([10, 11, 12, 20], [21], captured[3:], "stop")])

    class Engine:
        async def generate(self, request):
            prompt, response_ids, raw_routes, stop = next(chunks)
            assert request["prompt_token_ids"] == [prompt]
            return {
                "responses": ["response"],
                "response_ids": [response_ids],
                "response_logprobs": [[-0.1]],
                "stop_reasons": [stop],
                "routed_experts": [response_routes(raw_routes, len(response_ids))],
            }

    client = InferenceEngineClient(
        [Engine()], SimpleNamespace(decode=lambda *_args, **_kwargs: "response"), _make_min_cfg()
    )
    output = asyncio.run(client.generate({"prompt_token_ids": [[10, 11, 12]], "sampling_params": {"max_tokens": 2}}))
    response = output["routed_experts"][0]
    assert output["response_ids"] == [[20, 21]]
    assert response.dtype == (np.uint16 if expert_offset else np.uint8)
    np.testing.assert_array_equal(response, captured[2:])
    # Main's compact batches can carry only the local response rows even when
    # the global padded action window is longer.
    num_experts = 384 if expert_offset else 8
    routes = RoutedExpertRows((response,), response_len=3, num_experts=num_experts).materialize()
    assert routes.dtype == (torch.int16 if expert_offset else torch.uint8)
    if mode == "native":
        # The production mismatch probe requests native routing with sentinel rows.
        routes.zero_()
    sequences = torch.tensor([[0, 0, 0, 10, 11, 12, 20, 21, 0]])
    attention = (sequences != 0).long()
    positions = (attention.cumsum(-1) - 1).masked_fill(attention == 0, 0)
    controller = MegatronRouterReplay(range(3), recompute_enabled=False)
    controller.num_moe_layers_total, controller.topk = 3, 2
    wrapper = _make_wrapper(monkeypatch, packing=packing, controller=controller)
    wrapper.actor_module[0].config.num_moe_experts = num_experts

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
            assert selected[tokens == 21].tolist() == [[0, 1]]
        return torch.zeros(1)

    with controller.scoring_mode(mode) if mode == "router_replay" else nullcontext():
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
    collector = ProbeCollector.__new__(ProbeCollector)
    collector.cfg = SimpleNamespace(
        trainer=SimpleNamespace(policy=SimpleNamespace(megatron_config=SimpleNamespace(moe_router_replay=True)))
    )
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
