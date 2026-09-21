"""Align vLLM MoE routes with generated-token log-probabilities."""

import base64
from io import BytesIO

import numpy as np


def decode_openai_routes(encoded: str) -> np.ndarray:
    """Decode the `.npy` base64 route payload in vLLM's OpenAI response."""
    try:
        binary = base64.b64decode(encoded, validate=True)
        if not binary.startswith(b"\x93NUMPY"):
            raise ValueError("route payload is not a NumPy array")
        with BytesIO(binary) as stream:
            routes = np.load(stream, allow_pickle=False)
            if stream.read(1):
                raise ValueError("route payload has trailing bytes")
    except (ValueError, OSError, EOFError) as error:
        raise ValueError("invalid vLLM routed-expert payload") from error
    if routes.ndim != 3 or not np.issubdtype(routes.dtype, np.integer):
        raise ValueError("vLLM routed experts must be an integer [tokens, layers, topk] array")
    if routes.shape[1] == 0 or routes.shape[2] == 0:
        raise ValueError("vLLM routed experts must include layers and top-k choices")
    if routes.size and (routes.min() < 0 or routes.max() > np.iinfo(np.int16).max):
        raise ValueError("vLLM routed experts exceed the int16 training carrier")
    return routes


def response_routes(captured: np.ndarray | None, num_response_tokens: int) -> list[list[list[int]]] | None:
    """Select the route rows that predicted each generated token.

    vLLM returns prompt routes on the first sample, then one route row for
    each previously generated token on later samples. For R generated tokens,
    the last R rows therefore correspond to the R prediction positions.
    """
    if captured is None:
        return None
    if captured.ndim != 3 or not np.issubdtype(captured.dtype, np.integer):
        raise ValueError("vLLM routed experts must be an integer [tokens, layers, topk] array")
    if captured.shape[1] == 0 or captured.shape[2] == 0:
        raise ValueError("vLLM routed experts must include layers and top-k choices")
    if num_response_tokens > captured.shape[0]:
        raise ValueError(f"vLLM returned {captured.shape[0]} routed rows for {num_response_tokens} generated tokens")
    if num_response_tokens == 0:
        return []
    return captured[-num_response_tokens:].tolist()
