"""Causal alignment of vLLM route rows and generated tokens."""

import base64
import io

import numpy as np
import pytest
import torch

from skyrl_train.inference_engines.vllm.route_capture import response_routes
from skyrl_train.models.megatron_router_replay import dense_replay_targets
from skyrl_train.trajectory_runners.routed_experts import normalize_routed_experts


def test_response_routes_select_prediction_positions() -> None:
    # Three prompt tokens produce the first response token. The next decode
    # routes the first response token to produce the second one.
    captured = np.array([[[1, 2]], [[3, 4]], [[5, 6]], [[7, 8]]], dtype=np.uint16)

    assert response_routes(captured, 2) == [[[5, 6]], [[7, 8]]]
    assert response_routes(captured, 0) == []
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
