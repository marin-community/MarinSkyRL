"""Token-level engine and weight-drift diagnostic for matched generating weights."""

from __future__ import annotations

import hashlib
import json
import math

import torch

from skyrl_train.policy_version import PolicyVersionSegment, validate_policy_version_segments
from skyrl_train.utils.importance_ratio_diagnostics import ratio_statistics


def mismatch_decomposition_record(
    *,
    response_ids: list[list[int]],
    version_rows: list[list[PolicyVersionSegment]],
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    reference_logprobs: torch.Tensor,
    current_logprobs: torch.Tensor,
    consuming_step: int,
    reference_version: int,
    tis_cap: float,
    sample_rows: int,
) -> dict:
    """Decompose tokens from a frozen reference or the freshly published policy.

    The caller must establish that the frozen reference model and the inference engine
    both received the policy weights published as ``reference_version``. The
    consuming policy itself is also the generating trainer at age zero, before
    the next optimizer update. Older versions without a matched trainer scorer
    are counted but never scored against the wrong weights.
    """
    if consuming_step < 1 or reference_version < 0 or tis_cap <= 0 or sample_rows < 0:
        raise ValueError("invalid mismatch decomposition configuration")
    batch_size = len(response_ids)
    if (
        len(version_rows) != batch_size
        or sequences.shape != attention_mask.shape
        or sequences.ndim != 2
        or sequences.shape[0] != batch_size
    ):
        raise ValueError("mismatch decomposition rows are misaligned")
    shape = loss_mask.shape
    if any(
        tensor.shape != shape for tensor in (response_mask, behavior_logprobs, reference_logprobs, current_logprobs)
    ):
        raise ValueError("mismatch decomposition probabilities and masks are misaligned")

    selected = loss_mask.detach().cpu() > 0
    response = response_mask.detach().cpu() > 0
    ages = torch.full(shape, -1, dtype=torch.int32)
    reference_tokens = torch.zeros(shape, dtype=torch.bool)
    fresh_tokens = torch.zeros(shape, dtype=torch.bool)
    for row, (ids, segments) in enumerate(zip(response_ids, version_rows, strict=True)):
        length = len(ids)
        if length > shape[1] or not response[row, :length].all() or response[row, length:].any():
            raise ValueError("response token IDs do not align with the trainer response mask")
        if selected[row, length:].any():
            raise ValueError("selected token lies beyond its response IDs")
        response_start = sequences.shape[1] - shape[1]
        if not torch.equal(attention_mask[row, response_start:].detach().cpu().bool(), response[row]):
            raise ValueError("response mask does not align with the trainer attention mask")
        if not torch.equal(sequences[row, response_start : response_start + length].detach().cpu(), torch.tensor(ids)):
            raise ValueError("response token IDs do not align with trainer sequence IDs")
        validate_policy_version_segments(
            segments,
            response_length=length,
            require_known=True,
            required_mask=selected[row, :length].tolist(),
        )
        for segment in segments:
            version = segment["policy_version"]
            if version is None:
                continue
            age = consuming_step - version - 1
            if age < 0:
                raise ValueError("sampled policy version is newer than optimizer consumption")
            start = segment["start"]
            span = slice(start, start + segment["token_count"])
            ages[row, span] = age
            if version == reference_version:
                reference_tokens[row, span] = True
            elif age == 0:
                fresh_tokens[row, span] = True

    selected_reference = selected & reference_tokens
    selected_fresh = selected & fresh_tokens
    selected_matched = selected_reference | selected_fresh
    a = behavior_logprobs.detach().cpu().double()
    b = reference_logprobs.detach().cpu().double().clone()
    c = current_logprobs.detach().cpu().double()
    b[selected_fresh] = c[selected_fresh]
    for score in (a, b, c):
        if not torch.isfinite(score[selected_matched]).all():
            raise ValueError("nonfinite selected-token log probability in mismatch decomposition")
    engine = b - a
    stale = c - b
    combined = c - a

    def summarize(mask: torch.Tensor) -> dict:
        engine_values, stale_values, combined_values = engine[mask], stale[mask], combined[mask]
        if not engine_values.numel():
            return {"tokens": 0}
        cancellation = (engine_values.abs() + stale_values.abs() - combined_values.abs()).clamp_min(0)
        return {
            "tokens": int(engine_values.numel()),
            "engine": ratio_statistics(engine_values),
            "stale": ratio_statistics(stale_values),
            "combined": ratio_statistics(combined_values),
            "reconstruction_abs_max": (engine_values + stale_values - combined_values).abs().max().item(),
            "opposite_sign_fraction": ((engine_values * stale_values) < 0).double().mean().item(),
            "canceled_absolute_mean": cancellation.mean().item(),
            "tis_capped_fraction": (combined_values > math.log(tis_cap)).double().mean().item(),
        }

    summaries = {"all": summarize(selected_matched)}
    for age in sorted(set(ages[selected_matched].tolist())):
        summaries[f"age_{age}"] = summarize(selected_matched & (ages == age))
    positions = torch.arange(shape[1]).unsqueeze(0)
    lengths = response.sum(-1, keepdim=True)
    for name, position_mask in (
        ("first_256", positions < 256),
        ("last_256", positions >= lengths - 256),
        ("middle", (positions >= 256) & (positions < lengths - 256)),
    ):
        summaries[name] = summarize(selected_matched & position_mask)

    ranked_rows = sorted(
        (row for row in range(batch_size) if selected_matched[row].any()),
        key=lambda row: hashlib.sha256(json.dumps(response_ids[row]).encode()).digest(),
    )[:sample_rows]
    samples = []
    for row in ranked_rows:
        positions = torch.nonzero(selected_matched[row]).flatten().tolist()
        samples.append(
            {
                "row": row,
                "sequence_ids": sequences[row].detach().cpu().tolist(),
                "attention_mask": attention_mask[row].detach().cpu().tolist(),
                "response_ids": response_ids[row],
                "loss_mask": selected[row, : len(response_ids[row])].tolist(),
                "version_segments": version_rows[row],
                "positions": positions,
                "A_inference": a[row, positions].tolist(),
                "B_generating_trainer": b[row, positions].tolist(),
                "B_source": [
                    "frozen_reference" if selected_reference[row, position] else "fresh_consuming_policy"
                    for position in positions
                ],
                "C_consumer_trainer": c[row, positions].tolist(),
            }
        )
    record = {
        "schema_version": 2,
        "consuming_step": consuming_step,
        "published_version_to_completed_updates": "identity; age = consuming_step - version - 1",
        "reference_version": reference_version,
        "fresh_scored_version": consuming_step - 1,
        "tis_cap": tis_cap,
        "selected_tokens": int(selected.sum()),
        "reference_tokens": int(selected_reference.sum()),
        "fresh_version_tokens": int(selected_fresh.sum()),
        "matched_tokens": int(selected_matched.sum()),
        "other_version_tokens": int((selected & ~selected_matched).sum()),
        "mixed_version_responses": sum(len(row) > 1 for row in version_rows),
        "summaries": summaries,
        "samples": samples,
    }
    return record
