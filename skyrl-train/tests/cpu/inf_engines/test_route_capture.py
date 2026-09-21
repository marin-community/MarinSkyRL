"""Causal alignment of vLLM route rows and generated tokens."""

import base64
from io import BytesIO

import numpy as np
import pytest

from skyrl_train.inference_engines.vllm.route_capture import decode_openai_routes, response_routes


def _wire_payload(array: np.ndarray) -> str:
    buffer = BytesIO()
    np.save(buffer, array, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def test_decode_openai_routes_reads_vllm_numpy_wire_format() -> None:
    routes = np.arange(4 * 3 * 2, dtype=np.uint16).reshape(4, 3, 2)

    np.testing.assert_array_equal(decode_openai_routes(_wire_payload(routes)), routes)


@pytest.mark.parametrize("payload", ["not base64", base64.b64encode(b"not a numpy array").decode("ascii")])
def test_decode_openai_routes_rejects_invalid_payload(payload: str) -> None:
    with pytest.raises(ValueError, match="invalid vLLM routed-expert payload"):
        decode_openai_routes(payload)


def test_decode_openai_routes_rejects_non_integer_rows() -> None:
    with pytest.raises(ValueError, match="integer \\[tokens, layers, topk\\] array"):
        decode_openai_routes(_wire_payload(np.ones((2, 3, 4), dtype=np.float32)))


def test_decode_openai_routes_rejects_ids_that_would_wrap_in_training() -> None:
    with pytest.raises(ValueError, match="exceed the int16 training carrier"):
        decode_openai_routes(_wire_payload(np.asarray([[[32768]]], dtype=np.uint16)))


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
