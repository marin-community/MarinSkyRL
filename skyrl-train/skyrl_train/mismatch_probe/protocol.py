"""Deterministic identity and token-logprob primitives for mismatch archives."""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence


def request_seed(master_seed: int, prompt_id: str, repetition: int) -> int:
    """Derive a stable per-answer vLLM seed independent of Python hash salt."""
    payload = f"{master_seed}:{len(prompt_id)}:{prompt_id}:{repetition}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _feed_ints(hasher, values: Sequence[int]) -> None:
    hasher.update(struct.pack(">Q", len(values)))
    for value in values:
        hasher.update(struct.pack(">q", value))


def probe_hash(
    records: Iterable[tuple[str, Sequence[int], Sequence[int], Sequence[bool], Sequence[bool]]],
) -> str:
    """Hash ordered sample identities, exact token IDs, boundaries and masks."""
    hasher = hashlib.sha256(b"skyrl-mismatch-probe-v1\0")
    for sample_id, prompt_ids, response_ids, response_mask, loss_mask in records:
        encoded_id = sample_id.encode("utf-8")
        hasher.update(struct.pack(">Q", len(encoded_id)))
        hasher.update(encoded_id)
        for values in (prompt_ids, response_ids, response_mask, loss_mask):
            _feed_ints(hasher, values)
    return f"sha256:{hasher.hexdigest()}"


def require_token_identity(
    *,
    sample_id: str,
    expected_prompt: Sequence[int],
    trainer_prompt: Sequence[int],
    engine_response: Sequence[int],
    trainer_response: Sequence[int],
) -> None:
    """Fail before numerical analysis if scoring would compare different text."""
    if list(expected_prompt) != list(trainer_prompt):
        raise ValueError(f"mismatch probe token identity failed for {sample_id}: prompt IDs differ")
    if list(engine_response) != list(trainer_response):
        raise ValueError(f"mismatch probe token identity failed for {sample_id}: response IDs differ")


def chosen_logprobs_from_prompt_logprobs(
    *,
    prompt_lengths: Sequence[int],
    response_ids: Sequence[Sequence[int]],
    prompt_logprobs: Sequence[Sequence[dict[int, float] | None]],
) -> list[list[float]]:
    """Read exact chosen tokens at causal positions from vLLM prompt scoring."""
    if len(prompt_lengths) != len(response_ids) or len(response_ids) != len(prompt_logprobs):
        raise ValueError("vLLM prompt-logprob rows do not align with frozen probe rows")
    result = []
    for row, (prefix_length, tokens, scores) in enumerate(
        zip(prompt_lengths, response_ids, prompt_logprobs, strict=True)
    ):
        chosen = []
        for offset, token in enumerate(tokens):
            position = prefix_length + offset
            if position >= len(scores) or scores[position] is None or token not in scores[position]:
                raise ValueError(f"vLLM omitted frozen token {token} at row {row}, response offset {offset}")
            chosen.append(float(scores[position][token]))
        result.append(chosen)
    return result
