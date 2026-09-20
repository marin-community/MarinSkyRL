"""Load cached conversations as bounded teacher-forced EAGLE sequences."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import os
from typing import Any

import datasets
from loguru import logger
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class EagleReplaySequence:
    """One exact target sequence and the first token supervised by EAGLE."""

    token_ids: list[int]
    loss_start: int
    group_id: str


@dataclass(frozen=True)
class EagleReplaySelection:
    """A globally bounded replay selection."""

    sequences: tuple[EagleReplaySequence, ...]
    charged_tokens: int
    skipped_rows: int


def _token_ids(value: object, *, field: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list) or any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise ValueError(f"{field} must resolve to a list of integer token IDs")
    return value


def _messages(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("prompt must be a chat-message sequence")
    messages = []
    for message in value:
        if not isinstance(message, Mapping):
            raise ValueError("prompt messages must be mappings")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("prompt messages require string role and content fields")
        messages.append({"role": role, "content": content})
    return messages


def _text_sequence(
    prompt: object,
    response: object,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[int], int]:
    if not isinstance(response, str) or not response:
        raise ValueError("response must be a nonempty string")
    if isinstance(prompt, str):
        prompt_ids = _token_ids(tokenizer.encode(prompt, add_special_tokens=True), field="prompt")
        response_ids = _token_ids(tokenizer.encode(response, add_special_tokens=False), field="response")
        return [*prompt_ids, *response_ids], len(prompt_ids)

    messages = _messages(prompt)
    prompt_ids = _token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        ),
        field="prompt",
    )
    full_ids = _token_ids(
        tokenizer.apply_chat_template(
            [*messages, {"role": "assistant", "content": response}],
            tokenize=True,
            add_generation_prompt=False,
        ),
        field="prompt and response",
    )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("tokenizer chat template does not preserve the generation-prompt prefix")
    return full_ids, len(prompt_ids)


def replay_sequence_from_row(
    row: Mapping[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> EagleReplaySequence:
    """Normalize one exact-token or prompt/response corpus row."""
    prompt_token_ids = row.get("prompt_token_ids")
    response_token_ids = row.get("response_token_ids", row.get("completion_token_ids"))
    if prompt_token_ids is not None or response_token_ids is not None:
        if prompt_token_ids is None or response_token_ids is None:
            raise ValueError("exact replay rows require both prompt_token_ids and response_token_ids")
        prompt_ids = _token_ids(prompt_token_ids, field="prompt_token_ids")
        response_ids = _token_ids(response_token_ids, field="response_token_ids")
        token_ids = [*prompt_ids, *response_ids]
        loss_start = len(prompt_ids)
    else:
        prompt = row.get("prompt", row.get("input_prompt"))
        response = row.get("response", row.get("output_response"))
        token_ids, loss_start = _text_sequence(prompt, response, tokenizer)

    if not token_ids or loss_start <= 0 or loss_start >= len(token_ids):
        raise ValueError("EAGLE replay rows require at least one prompt and one response token")
    group_id = row.get("group_id")
    if group_id is None:
        prompt_digest = hashlib.sha256(bytes(str(token_ids[:loss_start]), "utf-8")).hexdigest()
        group_id = prompt_digest[:24]
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a nonempty string")
    return EagleReplaySequence(token_ids=token_ids, loss_start=loss_start, group_id=group_id)


def load_replay_rows(sources: Sequence[str]) -> Iterator[dict[str, Any]]:
    """Yield rows from JSON, JSONL, or Parquet corpus shards."""
    for source in sources:
        extension = os.path.splitext(source)[-1].lower()
        if extension == ".parquet":
            dataset = datasets.load_dataset("parquet", data_files=source, keep_in_memory=False, split="train")
        elif extension in {".json", ".jsonl"}:
            dataset = datasets.load_dataset("json", data_files=source, keep_in_memory=False, split="train")
        else:
            raise ValueError(f"EAGLE replay corpus must be JSONL or Parquet, got {source!r}")
        yield from dataset


def select_replay_sequences(
    rows: Iterator[Mapping[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_tokens: int,
    max_window_tokens: int,
    max_sequences_per_group: int,
) -> EagleReplaySelection:
    """Choose a deterministic prefix within the capture's global token budget."""
    selected = []
    charged_tokens = 0
    skipped_rows = 0
    group_counts: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        try:
            sequence = replay_sequence_from_row(row, tokenizer)
        except ValueError as error:
            logger.warning("Skipping malformed EAGLE replay row {}: {}", row_index, error)
            skipped_rows += 1
            continue
        if group_counts.get(sequence.group_id, 0) >= max_sequences_per_group:
            skipped_rows += 1
            continue
        charge = min(len(sequence.token_ids), max_window_tokens)
        if charged_tokens + charge > max_tokens:
            skipped_rows += 1
            continue
        selected.append(sequence)
        charged_tokens += charge
        group_counts[sequence.group_id] = group_counts.get(sequence.group_id, 0) + 1
        if charged_tokens == max_tokens:
            break
    return EagleReplaySelection(tuple(selected), charged_tokens, skipped_rows)


def replay_batches(
    selection: EagleReplaySelection,
    batch_size: int,
) -> Iterator[tuple[EagleReplaySequence, ...]]:
    """Yield large host batches so vLLM keeps every replay scheduler fed."""
    if batch_size <= 0:
        raise ValueError("EAGLE replay batch_size must be positive")
    for start in range(0, len(selection.sequences), batch_size):
        yield selection.sequences[start : start + batch_size]
