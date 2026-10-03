"""Decode vLLM expert routes onto exact generated token positions."""

import base64
import binascii
import io
import numpy as np


def decode_routed_experts(routes: str, expected_rows: int) -> np.ndarray:
    """Decode the complete vLLM route array, including prompt rows."""
    try:
        payload = base64.b64decode(routes, validate=True)
        if not payload.startswith(b"\x93NUMPY"):
            raise ValueError("missing NumPy array header")
        rows = np.load(io.BytesIO(payload), allow_pickle=False)
    except (binascii.Error, EOFError, ValueError, OSError) as error:
        raise ValueError("routed_experts must be a base64-encoded NumPy array") from error
    if rows.ndim != 3:
        raise ValueError("routed_experts must have [token, layer, expert] shape")
    if rows.shape[0] != expected_rows:
        raise ValueError(f"routed_experts has {rows.shape[0]} token rows; expected {expected_rows}")
    return rows


def normalize_routed_experts(routes: str, prompt_ids: list[int], response_ids: list[int]) -> np.ndarray:
    """Return response routes from vLLM's encoded route array, with a sentinel for the final unforwarded token."""
    rows = decode_routed_experts(routes, len(prompt_ids) + len(response_ids) - 1)
    return response_routes(generation_routes(rows, len(prompt_ids), len(response_ids)))


def generation_routes(rows: np.ndarray, prompt_length: int, response_length: int) -> np.ndarray:
    """Return the routes of the forward passes that generated each response token, in a compact unsigned dtype.

    vLLM's route array starts at the first prompt token and ends at the penultimate generated token, so its row
    ``prompt_length - 1 + i`` is the forward pass whose output was response token ``i``.
    """
    expected = prompt_length + response_length - 1
    if (
        prompt_length < 1
        or rows.ndim != 3
        or rows.shape[0] != expected
        or rows.shape[1] == 0
        or rows.shape[2] == 0
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.any(rows > np.iinfo(np.uint32).max)
    ):
        raise ValueError(
            f"routed_experts must be a nonnegative integer [token, layer, expert] array with {expected} token rows, "
            f"got shape {rows.shape} and dtype {rows.dtype}"
        )
    selected = rows[prompt_length - 1 :]
    dtype = np.uint8 if not selected.size or selected.max() <= 255 else np.min_scalar_type(int(selected.max()))
    return selected.astype(dtype)


def response_routes(generation_rows: np.ndarray) -> np.ndarray:
    """Return each response token's route from ``generation_routes`` rows.

    A response token's route is the forward pass that read it, which is the row that generated the next token. The
    final token is never read, so it gets an all-zero sentinel row.
    """
    result = np.empty_like(generation_rows)
    if len(generation_rows):
        result[:-1] = generation_rows[1:]
        result[-1] = 0
    return result
