"""Explicit Torch-to-NumPy transport at the MSRL learner boundary."""

from __future__ import annotations

import numpy as np
import torch

from skyrl_train.learner import LearnerBatch, LogProbResult, UpdateRequest
from skyrl_train.training_batch import GLOBAL_LOSS_DENOM_METADATA_KEY, TrainingInputBatch


BEHAVIOR_POLICY_VERSIONS_METADATA_KEY = "behavior_policy_versions"
OLD_POLICY_VERSION_METADATA_KEY = "old_policy_version"


def _numpy_copy(tensor: torch.Tensor, *, dtype: np.dtype | None = None) -> np.ndarray:
    array = tensor.detach().cpu().numpy().copy()
    return array.astype(dtype, copy=False) if dtype is not None else array


def learner_batch_from_training_input(
    training_input: TrainingInputBatch,
) -> LearnerBatch:
    """Copy an MSRL CPU Torch batch into the framework-neutral learner format."""
    metadata = training_input.metadata or {}
    versions = metadata.get(BEHAVIOR_POLICY_VERSIONS_METADATA_KEY)
    if versions is None:
        raise ValueError("learner batches require one receiver-observed behavior-policy version per row")
    return LearnerBatch(
        sequences=_numpy_copy(training_input["sequences"], dtype=np.dtype(np.int64)),
        attention_mask=_numpy_copy(training_input["attention_mask"]),
        response_mask=_numpy_copy(training_input["response_mask"]),
        loss_mask=_numpy_copy(training_input["loss_mask"], dtype=np.dtype(np.float32)),
        rollout_log_probs=(
            _numpy_copy(training_input["rollout_logprobs"], dtype=np.dtype(np.float32))
            if training_input.get("rollout_logprobs") is not None
            else None
        ),
        behavior_policy_versions=np.asarray(versions, dtype=np.int64),
    )


def apply_log_prob_result(training_input: TrainingInputBatch, result: LogProbResult) -> TrainingInputBatch:
    """Install learner log probabilities into the existing MSRL batch channels."""
    expected_shape = training_input["response_mask"].shape
    for name, values in (("policy", result.policy_log_probs), ("reference", result.reference_log_probs)):
        if values is not None and values.shape != expected_shape:
            raise ValueError(f"learner {name} log-probability shape {values.shape} must match {expected_shape}")
    training_input["action_log_probs"] = torch.from_numpy(result.policy_log_probs.copy())
    training_input["base_action_log_probs"] = (
        torch.from_numpy(result.reference_log_probs.copy()) if result.reference_log_probs is not None else None
    )
    training_input["values"] = None
    if training_input.metadata is None:
        training_input.metadata = {}
    training_input.metadata[OLD_POLICY_VERSION_METADATA_KEY] = result.policy_version
    return training_input


def update_request_from_training_input(
    training_input: TrainingInputBatch,
    *,
    global_step: int,
) -> UpdateRequest:
    """Copy the post-advantage MSRL batch into one whole learner update request."""
    metadata = training_input.metadata or {}
    old_policy_version = metadata.get(OLD_POLICY_VERSION_METADATA_KEY)
    if old_policy_version is None:
        raise ValueError("learner update requires the old-policy version from compute_log_probs")
    return UpdateRequest(
        batch=learner_batch_from_training_input(training_input),
        advantages=_numpy_copy(training_input["advantages"], dtype=np.dtype(np.float32)),
        old_policy_log_probs=_numpy_copy(training_input["action_log_probs"], dtype=np.dtype(np.float32)),
        old_policy_version=int(old_policy_version),
        reference_log_probs=(
            _numpy_copy(training_input["base_action_log_probs"], dtype=np.dtype(np.float32))
            if training_input.get("base_action_log_probs") is not None
            else None
        ),
        global_step=global_step,
        global_loss_denominator=metadata.get(GLOBAL_LOSS_DENOM_METADATA_KEY),
    )
