"""Causal alignment of vLLM route rows and generated tokens."""

import numpy as np
import pytest

from skyrl_train.inference_engines.vllm.route_capture import response_routes


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
