"""Turn closure-session evidence into FTPO alternatives and masked suffix repairs."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from rolloutengine.closure_repair import (
    ClosureTarget,
    ClosureTargetKind,
    verified_closure_candidates,
    verified_closure_target,
)
from rolloutengine.closure_scoring import closure_state, score_closure
from skyrl_gym.closure_task import ClosurePrompt
from skyrl_gym.verification import VerificationResult, VerificationStatus

from skyrl_train.ftpo import VerifiedFTPOBoundary, repeated_tool_opener_boundary


@dataclass(frozen=True)
class ClosureTrainingTargets:
    boundaries: list[VerifiedFTPOBoundary | None]
    repairs: list[ClosureTarget | None]


def closure_training_targets(
    prompt_ids: Sequence[Sequence[int]],
    response_ids: Sequence[Sequence[int]],
    loss_masks: Sequence[Sequence[int]],
    candidates: torch.Tensor,
    verifications: Sequence[VerificationResult | None],
    *,
    eligible: Sequence[bool],
    encode: Callable[[str], list[int]],
    decode: Callable[[list[int]], str],
    eos_token_ids: frozenset[int],
) -> ClosureTrainingTargets:
    """Use verified closure contracts; infrastructure failures produce no target.

    Repeated-opener targets select JSON-opening alternatives at the first
    repeated marker. Other families require a closing-only repair that preserves
    the task's required content. Missing suffixes are reserved for masked SFT.
    """
    boundaries = []
    repairs = []
    for row, (prompt, response, mask, verification, allowed) in enumerate(
        zip(prompt_ids, response_ids, loss_masks, verifications, eligible, strict=True)
    ):
        boundaries.append(None)
        repairs.append(None)
        if not allowed or verification is None or verification.status != VerificationStatus.VERIFIED:
            continue
        contract = ClosurePrompt.model_validate(
            {**verification.diagnostics["closure"], "prompt_token_ids": list(prompt)}
        )
        if len(mask) != len(response) or not all(mask):
            raise ValueError("Closure training requires a single unstitched model response")
        if contract.family == "repeated_opener_retention":
            position = repeated_tool_opener_boundary(prompt, response, mask, decode)
            if position is not None:
                chosen = tuple(token for token in candidates[row, position].tolist() if decode([token]).strip() == "{")
                boundaries[row] = VerifiedFTPOBoundary(position, chosen)
            continue
        content_response = response
        if response and response[-1] in eos_token_ids and verification.diagnostics.get("stop_reason") == "stop":
            content_response = response[:-1]
        target = verified_closure_target(
            prompt,
            content_response,
            structure_prefix=contract.prefix,
            inspect=lambda text: closure_state(text, contract.format),
            verify_complete=lambda text: score_closure(text, contract.expected, contract.format).success,
            encode=encode,
            decode=decode,
        )
        if target is None:
            continue
        if target.kind == ClosureTargetKind.MISSING_SUFFIX:
            repairs[row] = target
            continue
        chosen = verified_closure_candidates(
            target, prompt, response, candidates[row, target.response_position].tolist(), decode=decode
        )
        boundaries[row] = VerifiedFTPOBoundary(target.response_position, chosen)
    return ClosureTrainingTargets(boundaries, repairs)
