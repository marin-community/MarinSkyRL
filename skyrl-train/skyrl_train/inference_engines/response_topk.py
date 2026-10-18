"""Keep exact behavior-policy top-K IDs without relying on decoded token strings."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from typing import Any, NamedTuple, Protocol

import numpy as np


class FlatResponseLogprobs(Protocol):
    start_indices: list[int]
    end_indices: list[int]
    token_ids: list[int]
    logprobs: list[float]


class ResponseTopK(NamedTuple):
    sampled_logprobs: list[float]
    topk_indices: np.ndarray
    topk_logprobs: np.ndarray


def select_flat_response_topk(
    flat: FlatResponseLogprobs,
    response_ids: list[int],
    top_k: int,
    requested_ids: list[int] | None = None,
) -> ResponseTopK:
    """Select chosen-token and top-K full-vocabulary log probabilities.

    Each row contains the sampled token followed by the requested candidates.
    The sampled token may duplicate a head entry. Preserve the dictionary
    path's ordering for tied probabilities, including a tied sampled token.
    """
    stride = top_k + 1
    starts = np.arange(len(response_ids), dtype=np.int64) * stride
    if (
        top_k <= 0
        or len(flat.token_ids) != len(response_ids) * stride
        or len(flat.logprobs) != len(flat.token_ids)
        or not np.array_equal(flat.start_indices, starts)
        or not np.array_equal(flat.end_indices, starts + stride)
    ):
        raise ValueError("vLLM flat response logprobs do not match the requested candidate width")
    ids = np.asarray(flat.token_ids, dtype=np.int64).reshape(-1, stride)
    scores = np.asarray(flat.logprobs, dtype=np.float32).reshape(-1, stride)
    if not np.array_equal(ids[:, 0], response_ids):
        raise ValueError("vLLM flat sampled token IDs do not match the response")
    if np.any(ids < 0) or not np.isfinite(scores).all() or np.any(scores > 0):
        raise ValueError("vLLM flat response requires nonnegative IDs and finite nonpositive logprobs")
    sampled = scores[:, 0].tolist()
    if requested_ids is not None:
        if len(requested_ids) != top_k or not np.all(ids[:, 1:] == np.asarray(requested_ids)):
            raise ValueError("vLLM flat response omitted requested token IDs or changed their order")
        head_ids, head_scores = ids[:, 1:], scores[:, 1:]
    else:
        # Deduplicate only the extra sampled entry; the exact top-K itself is unique.
        order_scores = scores.copy()
        order_scores[:, 0] = np.where(np.any(ids[:, :1] == ids[:, 1:], axis=1), -np.inf, scores[:, 0])
        order = np.lexsort((ids, -order_scores), axis=-1)[:, :top_k]
        head_ids = np.take_along_axis(ids, order, axis=1)
        head_scores = np.take_along_axis(scores, order, axis=1)
    dtype = np.int32 if ids.size == 0 or ids.max() <= np.iinfo(np.int32).max else np.int64
    return ResponseTopK(sampled, np.ascontiguousarray(head_ids, dtype=dtype), np.ascontiguousarray(head_scores))


def select_response_topk(logprobs: Mapping[int, float], top_k: int) -> tuple[list[int], list[float]]:
    """Select the actual top K even when serving also reports the sampled token."""
    if top_k <= 0:
        raise ValueError("response top-K width must be positive")
    candidates = []
    for token_id, value in logprobs.items():
        if not isinstance(token_id, int) or token_id < 0:
            raise ValueError("response top-K requires exact non-negative token IDs")
        score = float(value)
        if not math.isfinite(score) or score > 0:
            raise ValueError("response top-K requires finite non-positive log probabilities")
        candidates.append((token_id, score))
    if len(candidates) < top_k:
        raise ValueError("vLLM returned fewer response candidates than the requested top-K")
    selected = sorted(candidates, key=lambda item: (-item[1], item[0]))[:top_k]
    return [token_id for token_id, _ in selected], [score for _, score in selected]


def select_chat_response_topk(items: list[dict[str, Any]], top_k: int) -> tuple[list[int], list[float]]:
    """Read vLLM's exact ``token_id:N`` chat representation, never decoded text."""
    scores = {}
    for item in items:
        token = item.get("token")
        match = re.fullmatch(r"token_id:([0-9]+)", token) if isinstance(token, str) else None
        if match is None:
            raise ValueError("chat response top-K omitted exact token IDs")
        token_id = int(match.group(1))
        if token_id in scores:
            raise ValueError("chat response top-K repeated a token ID")
        scores[token_id] = item.get("logprob")
    return select_response_topk(scores, top_k)
