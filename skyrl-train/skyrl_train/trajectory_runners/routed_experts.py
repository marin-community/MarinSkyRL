"""Decode vLLM expert routes onto exact generated token positions."""

import base64
import binascii
import io
from typing import Any

import numpy as np


def decode_routed_experts(routes: str) -> np.ndarray:
    """Decode vLLM's base64-encoded NumPy routes into a [token, layer, expert] array."""
    try:
        payload = base64.b64decode(routes, validate=True)
        if not payload.startswith(b"\x93NUMPY"):
            raise ValueError("missing NumPy array header")
        rows = np.load(io.BytesIO(payload), allow_pickle=False)
    except (binascii.Error, EOFError, ValueError, OSError) as error:
        raise ValueError("routed_experts must be a base64-encoded NumPy array") from error
    if rows.ndim != 3:
        raise ValueError("routed_experts must have [token, layer, expert] shape")
    return rows


def encode_routed_experts(rows: np.ndarray) -> str:
    """Encode routes the way vLLM returns them."""
    buffer = io.BytesIO()
    np.save(buffer, rows, allow_pickle=False)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def normalize_routed_experts(
    routes: Any, prompt_ids: list[int] | None, response_ids: list[int]
) -> list[list[list[int]]]:
    """Return response routes, with a sentinel for the final unforwarded token.

    vLLM's encoded array starts at the first prompt token and ends at the
    penultimate generated token. The last generated token has no forward pass.
    Already decoded per-response rows are accepted for in-process producers.
    """
    if isinstance(routes, str):
        if prompt_ids is None:
            raise ValueError("encoded routed_experts requires exact prompt token IDs")
        rows = decode_routed_experts(routes)
        expected = len(prompt_ids) + len(response_ids) - 1
        if rows.shape[0] != expected:
            raise ValueError(f"routed_experts has {rows.shape[0]} token rows; expected {expected}")
        rows = rows[len(prompt_ids) :]
        if response_ids:
            rows = np.concatenate((rows, np.zeros((1, *rows.shape[1:]), dtype=rows.dtype)))
    elif isinstance(routes, list):
        if len(routes) != len(response_ids):
            raise ValueError("routed_experts must align with exact response token IDs")
        try:
            rows = np.asarray(routes)
        except ValueError as error:
            raise ValueError("routed_experts must have [token, layer, expert] shape") from error
    else:
        raise ValueError("routed_experts must be a per-token list or base64-encoded NumPy array")

    if (
        rows.ndim != 3
        or rows.shape[1] == 0
        or rows.shape[2] == 0
        or not np.issubdtype(rows.dtype, np.integer)
        or np.any(rows < 0)
        or np.any(rows > np.iinfo(np.int16).max)
    ):
        raise ValueError("routed_experts must have [token, layer, expert] nonnegative int16 shape")
    return rows.tolist()
