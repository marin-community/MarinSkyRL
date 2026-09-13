"""Trainer-side tensors for context distillation.

The runner ships two prompts per edited sample: the training-context prompt (guidance
span removed) in ``prompt_token_ids`` and the served one in ``rollout_prompt_token_ids``.
This module builds the second padded sequence tensor, folds the deliberate context shift
out of the behaviour-logprob ratio, and reports how far apart the two contexts still are.
See :mod:`skyrl_train.trajectory_runners.context_distillation` for the mechanism.
"""

from __future__ import annotations

import torch
from loguru import logger

from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.trajectory_runners.context_distillation import ContextDistillationConfig

# TrainingInputBatch keys that exist only while the logprob phase needs them.
ROLLOUT_SEQUENCES_KEY = "rollout_sequences"
ROLLOUT_ATTENTION_MASK_KEY = "rollout_attention_mask"
CONTEXT_EDITED_TENSOR_KEY = "context_edited"
ROLLOUT_CONTEXT_TENSOR_KEYS = (ROLLOUT_SEQUENCES_KEY, ROLLOUT_ATTENTION_MASK_KEY, CONTEXT_EDITED_TENSOR_KEY)

CONTEXT_DISTILLATION_TRAINER_METRIC_PREFIX = "context_distillation/"
EDITED_FRACTION_METRIC = CONTEXT_DISTILLATION_TRAINER_METRIC_PREFIX + "edited_fraction"
# Mean over edited response tokens of log pi(y | rollout context) - log pi(y | training
# context): the per-token forward KL estimate on the sampled tokens. It is what the block
# still buys; distillation is working when it falls toward zero.
SHIFT_PER_TOKEN_METRIC = CONTEXT_DISTILLATION_TRAINER_METRIC_PREFIX + "shift_per_token"
SHIFT_ABS_PER_TOKEN_METRIC = CONTEXT_DISTILLATION_TRAINER_METRIC_PREFIX + "shift_abs_per_token"
SHIFT_PER_TRAJECTORY_METRIC = CONTEXT_DISTILLATION_TRAINER_METRIC_PREFIX + "shift_per_trajectory"
# The block acts mostly on the first assistant turn, so the shift is also reported over the first
# SHIFT_HEAD_TOKENS response positions and over the rest.
SHIFT_HEAD_TOKENS = 512
SHIFT_PER_TOKEN_HEAD_METRIC = CONTEXT_DISTILLATION_TRAINER_METRIC_PREFIX + "shift_per_token_head"
SHIFT_PER_TOKEN_TAIL_METRIC = CONTEXT_DISTILLATION_TRAINER_METRIC_PREFIX + "shift_per_token_tail"
SHIFT_METRIC_KEYS = (
    SHIFT_PER_TOKEN_METRIC,
    SHIFT_ABS_PER_TOKEN_METRIC,
    SHIFT_PER_TRAJECTORY_METRIC,
    SHIFT_PER_TOKEN_HEAD_METRIC,
    SHIFT_PER_TOKEN_TAIL_METRIC,
)
CONTEXT_SHIFT_METRIC_KEYS = (EDITED_FRACTION_METRIC, *SHIFT_METRIC_KEYS)


def build_rollout_context_tensors(
    tokenizer,
    rollout_prompt_token_ids: list[list[int]],
    response_ids: list[list[int]],
    rewards,
    loss_masks: list[list[int]],
    response_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad the served prompts with the same responses into ``(sequences, attention_mask)``.

    The prompt axis is padded on its own (the served prompts are longer), the response
    axis must come out identical to the training batch so both forwards slice the same
    ``response_length`` positions.
    """
    sequences, attention_mask, rollout_response_mask, *_ = convert_prompts_responses_to_batch_tensors(
        tokenizer, rollout_prompt_token_ids, response_ids, rewards, loss_masks
    )
    if not torch.equal(rollout_response_mask, response_mask):
        raise ValueError("rollout-context batch disagrees with the training batch on the response axis")
    return sequences, attention_mask


def attach_rollout_context(
    training_input: TrainingInputBatch,
    rollout_sequences: torch.Tensor,
    rollout_attention_mask: torch.Tensor,
    context_edited: list[bool],
) -> None:
    training_input[ROLLOUT_SEQUENCES_KEY] = rollout_sequences
    training_input[ROLLOUT_ATTENTION_MASK_KEY] = rollout_attention_mask
    training_input[CONTEXT_EDITED_TENSOR_KEY] = torch.tensor(context_edited, dtype=torch.bool)


def has_rollout_context(training_input: TrainingInputBatch) -> bool:
    return all(key in training_input and training_input[key] is not None for key in ROLLOUT_CONTEXT_TENSOR_KEYS)


def rollout_context_forward_batch(
    training_input: TrainingInputBatch, forward_batch: TrainingInputBatch
) -> TrainingInputBatch | None:
    """The forward batch with the rollout-context sequences swapped in.

    Returns None when the batch carries no rollout context or no row was edited (every
    rollout prompt then equals its training prompt and the forward would be redundant).
    Every other key and the metadata are copied from ``forward_batch`` so replayed
    routing and ``response_length`` travel with the sequences.
    """
    if not has_rollout_context(training_input) or not bool(training_input[CONTEXT_EDITED_TENSOR_KEY].any()):
        return None
    tensors = {key: forward_batch[key] for key in forward_batch}
    tensors["sequences"] = training_input[ROLLOUT_SEQUENCES_KEY]
    tensors["attention_mask"] = training_input[ROLLOUT_ATTENTION_MASK_KEY]
    batch = TrainingInputBatch(tensors)
    batch.metadata = dict(forward_batch.metadata)
    return batch


def rebase_behavior_logprobs(
    rollout_logprobs: torch.Tensor,
    training_context_logprobs: torch.Tensor,
    rollout_context_logprobs: torch.Tensor,
    context_edited: torch.Tensor,
) -> torch.Tensor:
    """Express behaviour logprobs relative to the training context.

    The loss forms ``exp(old - behaviour)`` where ``old`` is now ``log pi(y | training
    context)`` while the engine sampled from ``pi_engine(y | rollout context)``. Adding
    ``log pi(y | training) - log pi(y | rollout)`` to the behaviour logprobs of edited rows
    turns that ratio into ``exp(log pi(y | rollout) - log pi_engine(y | rollout))``: the
    engine-vs-trainer mismatch alone. The deliberate context shift never enters the
    importance weight, which is the P²O expectation over rollouts of the prompted policy.
    Unedited rows are returned unchanged.
    """
    edited = context_edited.to(rollout_logprobs.device)[:, None].to(rollout_logprobs.dtype)
    delta = (training_context_logprobs - rollout_context_logprobs).to(rollout_logprobs.dtype)
    return rollout_logprobs + delta * edited


def neutralize_behavior_logprobs(
    rollout_logprobs: torch.Tensor,
    training_context_logprobs: torch.Tensor,
    context_edited: torch.Tensor,
) -> torch.Tensor:
    """``tis_reference: none``: edited rows get a behaviour ratio of exactly one."""
    edited = context_edited.to(rollout_logprobs.device)[:, None]
    return torch.where(edited, training_context_logprobs.to(rollout_logprobs.dtype), rollout_logprobs)


def context_shift_metrics(
    training_context_logprobs: torch.Tensor,
    rollout_context_logprobs: torch.Tensor,
    loss_mask: torch.Tensor,
    context_edited: torch.Tensor,
) -> dict[str, float]:
    """How much the block still changes the policy on the tokens it produced.

    All means are over the loss-masked response tokens of edited rows. A batch with no edited
    row reports NaN for every shift key (there is nothing to measure; zero would read as
    "fully absorbed"), with the key set stable across steps.
    """
    edited = context_edited.to(loss_mask.device)
    metrics = dict.fromkeys(SHIFT_METRIC_KEYS, float("nan"))
    token_mask = (loss_mask > 0) & edited[:, None]
    n_tokens = float(token_mask.sum().item())
    n_rows = float(edited.sum().item())
    if n_tokens == 0.0 or n_rows == 0.0:
        return metrics
    shift = (rollout_context_logprobs - training_context_logprobs).float() * token_mask
    metrics[SHIFT_PER_TOKEN_METRIC] = float(shift.sum().item() / n_tokens)
    metrics[SHIFT_ABS_PER_TOKEN_METRIC] = float(shift.abs().sum().item() / n_tokens)
    metrics[SHIFT_PER_TRAJECTORY_METRIC] = float(shift.sum().item() / n_rows)
    head = token_mask.clone()
    head[:, SHIFT_HEAD_TOKENS:] = False
    tail = token_mask & ~head
    for key, region in ((SHIFT_PER_TOKEN_HEAD_METRIC, head), (SHIFT_PER_TOKEN_TAIL_METRIC, tail)):
        n_region = float(region.sum().item())
        if n_region > 0.0:
            metrics[key] = float((shift * region).sum().item() / n_region)
    return metrics


def apply_context_distillation_references(
    training_input: TrainingInputBatch,
    *,
    training_context_logprobs: torch.Tensor,
    rollout_context_logprobs: torch.Tensor | None,
    config: ContextDistillationConfig,
) -> dict[str, float]:
    """Re-base the behaviour logprobs of edited rows and drop the rollout-context tensors.

    Called once per step after the logprob phase. Returns the driver-side metrics; the
    batch leaves with the same keys it would have without the feature so the training
    dispatch stays lean and key-identical. The batch's own columns decide what happens:
    edited rows are re-based when the rollout-context logprobs are available, and
    neutralised (ratio one) otherwise, so a stripped prompt never trains against the
    prompted behaviour logprobs, whatever the driver's config says.
    """
    if not has_rollout_context(training_input):
        return {}
    context_edited = training_input[CONTEXT_EDITED_TENSOR_KEY]
    loss_mask = training_input["loss_mask"]
    # padded rows (pad_batch) are never edited; report the fraction over the real rows
    n_real = max(int(context_edited.numel()) - int((training_input.metadata or {}).get("pad_size", 0)), 1)
    metrics: dict[str, float] = {EDITED_FRACTION_METRIC: float(context_edited.sum().item()) / n_real}
    metrics.update(dict.fromkeys(SHIFT_METRIC_KEYS, float("nan")))
    if rollout_context_logprobs is not None:
        metrics.update(
            context_shift_metrics(training_context_logprobs, rollout_context_logprobs, loss_mask, context_edited)
        )
    rollout_logprobs = training_input.get("rollout_logprobs")
    if rollout_logprobs is not None and bool(context_edited.any()):
        if rollout_context_logprobs is not None:
            training_input["rollout_logprobs"] = rebase_behavior_logprobs(
                rollout_logprobs, training_context_logprobs, rollout_context_logprobs, context_edited
            )
        else:
            if config.rollout_context_forward_needed:
                logger.warning(
                    "context distillation: edited rows without rollout-context logprobs; "
                    "their behaviour ratio is neutralised for this step (no engine-mismatch correction)"
                )
            training_input["rollout_logprobs"] = neutralize_behavior_logprobs(
                rollout_logprobs, training_context_logprobs, context_edited
            )
    for key in ROLLOUT_CONTEXT_TENSOR_KEYS:
        training_input.pop(key, None)
    return metrics
