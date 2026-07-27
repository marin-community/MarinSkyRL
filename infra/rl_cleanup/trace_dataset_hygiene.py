"""Trace-dataset hygiene helpers.

Ported verbatim from the upstream trace-export tooling. These are load-bearing data
fixes, not cosmetics: surrogate code points make PyArrow refuse the dataset, bash
warning lines corrupt otherwise-valid records, and ShareGPT consumers require a
specific conversation shape. Do not "simplify" them without a reproducing dataset.
"""

from __future__ import annotations

import re
from typing import Any, Optional


_BASH_JOB_CONTROL_WARNING = "bash: initialize_job_control: no job control in background: Bad file descriptor"


def _clean_bash_warning_value(value):
    """Recursively clean bash job control warnings from a value.

    Defined at module level (not nested) so it can be pickled by HF datasets.
    """
    if isinstance(value, str):
        cleaned = value.replace(f"{_BASH_JOB_CONTROL_WARNING}\n", "")
        cleaned = cleaned.replace(f"\n{_BASH_JOB_CONTROL_WARNING}", "")
        return cleaned.replace(_BASH_JOB_CONTROL_WARNING, "")
    if isinstance(value, list):
        return [_clean_bash_warning_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _clean_bash_warning_value(val) for key, val in value.items()}
    return value


def _sanitize_bash_warning_record(record):
    """Sanitize a single record by cleaning bash warnings from all values.

    Defined at module level (not nested) so it can be pickled by HF datasets.
    """
    return {key: _clean_bash_warning_value(val) for key, val in record.items()}


def _sanitize_bash_warnings(dataset):
    """Strip bash job control warnings from trace datasets to avoid agent confusion."""
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return dataset

    if isinstance(dataset, DatasetDict):
        return DatasetDict({k: _sanitize_bash_warnings(v) for k, v in dataset.items()})
    if isinstance(dataset, Dataset):
        return dataset.map(_sanitize_bash_warning_record, load_from_cache_file=False)
    return dataset


def _finalize_trace_dataset(dataset):
    """Apply final cleanup/formatting before returning a dataset."""
    dataset = _sanitize_bash_warnings(dataset)
    dataset = _sanitize_surrogates(dataset)
    dataset = _ensure_sharegpt_conversations(dataset)
    return dataset


_SURROGATE_PATTERN = re.compile(r"[\ud800-\udfff]")


def _sanitize_surrogates(dataset):
    """Replace Unicode surrogate code points with spaces to keep PyArrow happy."""
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return dataset

    if isinstance(dataset, DatasetDict):
        return DatasetDict(
            {k: v.map(_strip_surrogates_from_record, load_from_cache_file=False) for k, v in dataset.items()}
        )
    if isinstance(dataset, Dataset):
        return dataset.map(_strip_surrogates_from_record, load_from_cache_file=False)
    return dataset


def _strip_surrogates_from_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _strip_surrogates(value) for key, value in record.items()}


def _strip_surrogates(value: Any) -> Any:
    if isinstance(value, str):
        return _SURROGATE_PATTERN.sub(" ", value)
    if isinstance(value, list):
        return [_strip_surrogates(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_surrogates(val) for key, val in value.items()}
    return value


def _ensure_sharegpt_conversations(dataset):
    """Guarantee conversation columns conform to ShareGPT expectations."""
    try:
        from datasets import Dataset, DatasetDict
    except Exception:
        return dataset

    def _map_record(record):
        conversations = record.get("conversations")
        if not isinstance(conversations, list):
            return record
        return {"conversations": _squash_system_turns(conversations)}

    if isinstance(dataset, DatasetDict):
        return DatasetDict({split: ds.map(_map_record, load_from_cache_file=False) for split, ds in dataset.items()})
    if isinstance(dataset, Dataset):
        return dataset.map(_map_record, load_from_cache_file=False)
    return dataset


def _squash_system_turns(conversations):
    """
    Merge consecutive system messages into user turns so the final transcript alternates
    user/assistant as expected by ShareGPT loaders.
    """
    if not isinstance(conversations, list):
        return conversations

    cleaned: list[dict[str, Any]] = []
    system_buffer: list[str] = []

    def _drain_buffer() -> Optional[str]:
        nonlocal system_buffer
        if not system_buffer:
            return None
        text = "\n\n".join(piece for piece in system_buffer if piece)
        system_buffer = []
        return text

    def _merge_buffer_with_user(content: str) -> str:
        prefix = _drain_buffer()
        if not prefix:
            return content
        if content:
            return f"{prefix}\n\n{content}"
        return prefix

    for message in conversations:
        role = message.get("role")
        content = message.get("content") or ""

        if role == "system":
            system_buffer.append(content)
            continue

        if role == "user":
            merged_content = _merge_buffer_with_user(content)
            cleaned.append({**message, "role": "user", "content": merged_content})
            continue

        buffered = _drain_buffer()
        if buffered:
            cleaned.append({"role": "user", "content": buffered})
        cleaned.append(dict(message))

    leftover = _drain_buffer()
    if leftover:
        if cleaned and cleaned[-1].get("role") == "user":
            existing = cleaned[-1].get("content") or ""
            cleaned[-1]["content"] = f"{existing}\n\n{leftover}" if existing else leftover
        else:
            cleaned.append({"role": "user", "content": leftover})

    if cleaned and cleaned[0].get("role") != "user":
        cleaned.insert(0, {"role": "user", "content": ""})

    return cleaned
