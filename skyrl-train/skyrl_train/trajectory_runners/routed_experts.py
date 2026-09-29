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
    """Return response routes, with a sentinel for the final unforwarded token.

    vLLM's encoded array starts at the first prompt token and ends at the
    penultimate generated token. The last generated token has no forward pass.
    """
    expected = len(prompt_ids) + len(response_ids) - 1
    rows = decode_routed_experts(routes, expected)
    if (
        rows.ndim != 3
        or rows.shape[1] == 0
        or rows.shape[2] == 0
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.any(rows > np.iinfo(np.uint32).max)
    ):
        raise ValueError("routed_experts must have [token, layer, expert] nonnegative integer shape")
    response_rows = rows[len(prompt_ids) :]
    dtype = (
        np.uint8
        if not response_rows.size or response_rows.max() <= 255
        else np.min_scalar_type(int(response_rows.max()))
    )
    result = np.empty((len(response_ids), *rows.shape[1:]), dtype=dtype)
    if not response_ids:
        return result
    result[:-1] = response_rows
    result[-1] = 0
    return result
