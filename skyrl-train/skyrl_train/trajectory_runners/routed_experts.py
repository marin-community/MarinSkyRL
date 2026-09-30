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


def _validated_route_rows(routes: str, prompt_ids: list[int], response_ids: list[int]) -> np.ndarray:
    rows = decode_routed_experts(routes, len(prompt_ids) + len(response_ids) - 1)
    if (
        rows.ndim != 3
        or rows.shape[1] == 0
        or rows.shape[2] == 0
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.any(rows > np.iinfo(np.uint32).max)
    ):
        raise ValueError("routed_experts must have [token, layer, expert] nonnegative integer shape")
    return rows


def _compact(rows: np.ndarray) -> np.ndarray:
    dtype = np.uint8 if not rows.size or rows.max() <= 255 else np.min_scalar_type(int(rows.max()))
    return rows.astype(dtype, copy=False)


def normalize_routed_experts(routes: str, prompt_ids: list[int], response_ids: list[int]) -> np.ndarray:
    """Return response routes, with a sentinel for the final unforwarded token.

    vLLM's encoded array starts at the first prompt token and ends at the
    penultimate generated token. The last generated token has no forward pass.
    """
    response_rows = _compact(_validated_route_rows(routes, prompt_ids, response_ids)[len(prompt_ids) :])
    result = np.empty((len(response_ids), *response_rows.shape[1:]), dtype=response_rows.dtype)
    if not response_ids:
        return result
    result[:-1] = response_rows
    result[-1] = 0
    return result


def prompt_routed_experts(routes: str, prompt_ids: list[int], response_ids: list[int]) -> np.ndarray:
    """Return the experts vLLM selected for each prompt token, one row per prompt token."""
    return _compact(_validated_route_rows(routes, prompt_ids, response_ids)[: len(prompt_ids)])
