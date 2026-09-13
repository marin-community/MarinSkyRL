"""Context distillation for prompt-augmented rollouts (P²O, arXiv 2603.21877).

A rollout can be sampled under a guidance block that the task harness appends to the
task instruction, so the first user message reads
``<instruction><start_marker><block><end_marker><terminal state>...``. Context
distillation keeps those rollouts but computes the policy gradient on the *training
context*: the same first user message with the span from ``start_marker`` up to
``end_marker`` removed. The model is trained to produce the block-conditioned behaviour
without the block, which is what it sees at evaluation.

The trainer re-tokenizes the initial prompt from the message text, so the edit is a text
edit made before that tokenization. Served response ids, loss masks and behaviour
logprobs are untouched: the block lives only in the prompt. The trainer never learns the
block text; it knows the two anchors only, so per-task or per-rollout blocks (an evolved
prompt per hard task) need no trainer change.

Two references have to follow the edit, both configured in
``trainer.algorithm.context_distillation``:

* the behaviour (TIS) reference. The rollout engine sampled under the *rollout context*,
  so the importance ratio that corrects engine-vs-trainer mismatch must compare logprobs
  computed under that same context (``tis_reference: rollout``, one extra no-grad policy
  forward per step) or be switched off for edited samples (``none``);
* the frozen reference for the KL term. ``rollout`` makes the KL pull the bare policy
  toward the block-conditioned reference (the teacher); ``training`` keeps the plain
  self-reference.

The tensor side (rollout-context sequences, re-basing behaviour logprobs, the shift
metrics) lives in :mod:`skyrl_train.utils.context_distillation`.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from skyrl_train.metric_names import (
    CONTEXT_DISTILLATION_ABSENT_METRIC,
    CONTEXT_DISTILLATION_EDITED_METRIC,
    CONTEXT_DISTILLATION_FAILED_METRIC,
    CONTEXT_DISTILLATION_REMOVED_TOKENS_METRIC,
)

# terminus-2 renders the first user message as ``... Task Description:\n{instruction}\n\n
# Current terminal state:\n...``; the guidance block is a strict suffix of the instruction
# behind this delimiter, so the span to remove ends where the terminal state begins.
DEFAULT_START_MARKER = "\n\n---\nWorking guidance:\n"
DEFAULT_END_MARKER = "\n\nCurrent terminal state:"
DEFAULT_MAX_REMOVED_CHARS = 4096

ON_FAILURE_CHOICES = ("mask", "error")
TIS_REFERENCE_CHOICES = ("rollout", "none")
KL_REFERENCE_CHOICES = ("rollout", "training")

# Per-sample TrajectoryBatch columns attached by the runner when the feature is on.
ROLLOUT_PROMPT_TOKEN_IDS_KEY = "rollout_prompt_token_ids"
CONTEXT_EDITED_KEY = "context_edited"
CONTEXT_DISTILLATION_ROW_KEYS = (ROLLOUT_PROMPT_TOKEN_IDS_KEY, CONTEXT_EDITED_KEY)


@dataclass(frozen=True)
class ContextDistillationConfig:
    """Validated view of ``trainer.algorithm.context_distillation``."""

    enabled: bool = False
    start_marker: str = DEFAULT_START_MARKER
    end_marker: str = DEFAULT_END_MARKER
    # A guidance block is a few hundred tokens; a longer removed span means the end marker was found
    # past something else the harness appended after the block (MCP servers, skills) and the edit is
    # refused rather than silently widening the strip.
    max_removed_chars: int = DEFAULT_MAX_REMOVED_CHARS
    on_failure: str = "error"
    tis_reference: str = "rollout"
    kl_reference: str = "training"

    @classmethod
    def disabled(cls) -> ContextDistillationConfig:
        return cls()

    @classmethod
    def from_algorithm_config(cls, algorithm_cfg: Mapping[str, Any] | None) -> ContextDistillationConfig:
        """Read the block from ``trainer.algorithm``; an absent block means disabled."""
        block = algorithm_cfg.get("context_distillation", None) if algorithm_cfg is not None else None
        if block is None:
            return cls.disabled()
        config = cls(
            enabled=bool(block.get("enabled", False)),
            start_marker=str(block.get("start_marker", DEFAULT_START_MARKER)),
            end_marker=str(block.get("end_marker", DEFAULT_END_MARKER)),
            max_removed_chars=int(block.get("max_removed_chars", DEFAULT_MAX_REMOVED_CHARS)),
            on_failure=str(block.get("on_failure", "error")),
            tis_reference=str(block.get("tis_reference", "rollout")),
            kl_reference=str(block.get("kl_reference", "training")),
        )
        config.validate()
        return config

    def validate(self) -> None:
        prefix = "trainer.algorithm.context_distillation"
        if self.on_failure not in ON_FAILURE_CHOICES:
            raise ValueError(f"{prefix}.on_failure must be one of {ON_FAILURE_CHOICES}, got {self.on_failure!r}")
        if self.tis_reference not in TIS_REFERENCE_CHOICES:
            raise ValueError(
                f"{prefix}.tis_reference must be one of {TIS_REFERENCE_CHOICES}, got {self.tis_reference!r}"
            )
        if self.kl_reference not in KL_REFERENCE_CHOICES:
            raise ValueError(f"{prefix}.kl_reference must be one of {KL_REFERENCE_CHOICES}, got {self.kl_reference!r}")
        if self.enabled and not self.start_marker:
            raise ValueError(f"{prefix}.start_marker must be non-empty when the feature is enabled")
        if self.max_removed_chars <= 0:
            raise ValueError(f"{prefix}.max_removed_chars must be positive, got {self.max_removed_chars}")

    @property
    def rollout_context_forward_needed(self) -> bool:
        """True when edited samples need a policy forward under the rollout context.

        Deliberately independent of ``enabled``: the batch's own columns say whether any row was
        edited (a group buffered under a previous config still carries stripped prompts), and a
        stripped row must be re-based whatever the driver's flag says today.
        """
        return self.tis_reference == "rollout"

    @property
    def reference_uses_rollout_context(self) -> bool:
        return self.kl_reference == "rollout"


class ContextEditStatus(StrEnum):
    ABSENT = "absent"  # no start marker: an unprompted rollout, trained as is
    STRIPPED = "stripped"  # the span was removed from the first user message
    START_REPEATED = "start_repeated"  # the start marker occurs more than once
    END_MISSING = "end_missing"  # no end marker after the start marker
    SPAN_TOO_LONG = "span_too_long"  # the span exceeds max_removed_chars: something else sits between the markers


FAILED_EDIT_STATUSES = frozenset(
    {ContextEditStatus.START_REPEATED, ContextEditStatus.END_MISSING, ContextEditStatus.SPAN_TOO_LONG}
)


@dataclass(frozen=True)
class ContextEdit:
    """Outcome of stripping the guidance span from one first user message."""

    status: ContextEditStatus
    text: str
    removed_chars: int = 0

    @property
    def stripped(self) -> bool:
        return self.status is ContextEditStatus.STRIPPED

    @property
    def failed(self) -> bool:
        return self.status in FAILED_EDIT_STATUSES


class ContextEditError(ValueError):
    """Raised for a failed edit when ``on_failure: error``."""


def strip_guidance_suffix(
    content: str, start_marker: str, end_marker: str, max_removed_chars: int | None = None
) -> ContextEdit:
    """Remove ``[start_marker, end_marker)`` from ``content``.

    The start marker must occur exactly once and the end marker must follow it; an empty
    end marker strips to the end of the text. A message without the start marker is an
    unprompted rollout and comes back unchanged with status ``ABSENT``. Failures return
    the original text with a typed status so the caller decides between masking the
    sample and raising; a span longer than ``max_removed_chars`` is a failure too, since
    the end marker is a template anchor and anything the harness appended after the block
    would otherwise be removed with it.
    """
    if not start_marker:
        raise ValueError("start_marker must be non-empty")
    occurrences = content.count(start_marker)
    if occurrences == 0:
        return ContextEdit(ContextEditStatus.ABSENT, content)
    if occurrences > 1:
        return ContextEdit(ContextEditStatus.START_REPEATED, content)
    start = content.index(start_marker)
    if end_marker:
        end = content.find(end_marker, start + len(start_marker))
        if end < 0:
            return ContextEdit(ContextEditStatus.END_MISSING, content)
    else:
        end = len(content)
    if max_removed_chars is not None and end - start > max_removed_chars:
        return ContextEdit(ContextEditStatus.SPAN_TOO_LONG, content, removed_chars=end - start)
    return ContextEdit(ContextEditStatus.STRIPPED, content[:start] + content[end:], removed_chars=end - start)


def context_distillation_batch_fields(outputs: Iterable[Any]) -> tuple[dict[str, list[Any]], dict[str, float]]:
    """Per-sample batch columns and per-batch counters for a runner's outputs.

    Each output carries ``evidence.prompt_token_ids`` (the training-context prompt),
    ``rollout_prompt_token_ids`` (the served prompt, set only when the span was stripped),
    ``context_edit`` and ``disposition.loss_eligible``. A sample counts as edited only while
    it is still trainable: a masked sample keeps its training prompt as the rollout prompt
    so both sequence tensors stay consistent with its (zeroed) loss mask.
    """
    rollout_prompts: list[list[int]] = []
    edited_flags: list[bool] = []
    counts = {
        CONTEXT_DISTILLATION_EDITED_METRIC: 0.0,
        CONTEXT_DISTILLATION_FAILED_METRIC: 0.0,
        CONTEXT_DISTILLATION_ABSENT_METRIC: 0.0,
        CONTEXT_DISTILLATION_REMOVED_TOKENS_METRIC: 0.0,
    }
    for output in outputs:
        training_prompt = list(output.evidence.prompt_token_ids)
        rollout_prompt: Sequence[int] | None = getattr(output, "rollout_prompt_token_ids", None)
        edit: ContextEdit | None = getattr(output, "context_edit", None)
        edited = rollout_prompt is not None and bool(output.disposition.loss_eligible)
        rollout_prompts.append(list(rollout_prompt) if edited else training_prompt)
        edited_flags.append(edited)
        if edited:
            counts[CONTEXT_DISTILLATION_EDITED_METRIC] += 1
            counts[CONTEXT_DISTILLATION_REMOVED_TOKENS_METRIC] += len(rollout_prompt) - len(training_prompt)
        elif edit is not None and edit.failed:
            counts[CONTEXT_DISTILLATION_FAILED_METRIC] += 1
        else:
            counts[CONTEXT_DISTILLATION_ABSENT_METRIC] += 1
    return {ROLLOUT_PROMPT_TOKEN_IDS_KEY: rollout_prompts, CONTEXT_EDITED_KEY: edited_flags}, counts
