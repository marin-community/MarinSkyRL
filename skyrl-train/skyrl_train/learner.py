"""Framework-neutral learner contract used by MSRL orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import math
from typing import Mapping, Protocol

import numpy as np


class PolicyLoss(StrEnum):
    REGULAR = "regular"
    BEHAVIOR_CLIP = "behavior_clip"


class LossNormalization(StrEnum):
    TOKEN_MEAN = "token_mean"
    SEQUENCE_MEAN = "sequence_mean"
    SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED = "seq_mean_token_sum_norm"
    GLOBAL_SEQUENCE_MEAN_TOKEN_SUM_NORMALIZED = "seq_mean_token_sum_norm_global"


class UpdateStatus(StrEnum):
    SUCCEEDED = "succeeded"
    SKIPPED = "skipped"


class PublicationStatus(StrEnum):
    NOT_STARTED = "not_started"
    OUTDATED = "outdated"
    PENDING = "pending"
    INSTALLED = "installed"
    FAILED = "failed"


class LearnerLifecycle(StrEnum):
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    FAILED = "failed"
    CLOSED = "closed"


class UnsupportedLearnerConfiguration(ValueError):
    """Raised before learner allocation when MSRL requests unsupported behavior."""


class LearnerPublicationIncomplete(RuntimeError):
    """Raised when inference has not installed the requested policy version."""


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise UnsupportedLearnerConfiguration(f"{name} must be finite")


def _require_shape(name: str, value: np.ndarray, shape: tuple[int, ...]) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} shape {value.shape} must match {shape}")


def _require_finite_array(name: str, value: np.ndarray | None) -> None:
    if value is not None and not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class LearnerConfig:
    """The GRPO semantics required from a learner."""

    policy_loss: PolicyLoss
    loss_normalization: LossNormalization
    requires_reference_log_probs: bool
    clip_low: float
    clip_high: float
    dual_clip_ratio: float
    reference_kl_coefficient: float | None
    kl_estimator_type: str
    use_absolute_kl: bool
    use_rollout_importance_sampling: bool
    rollout_importance_ratio_cap: float
    update_epochs: int
    logprob_temperature: float
    max_sequence_length: int
    require_rollout_logprobs: bool = False
    offpolicy_mask_enabled: bool = False
    offpolicy_mask_ratio: str = "mismatch"
    offpolicy_mask_low: float = 0.5
    offpolicy_mask_high: float = 5.0
    offpolicy_mask_veto_ratio: float = 1.0e-5
    offpolicy_mask_renormalize: bool = False

    def __post_init__(self) -> None:
        finite_values = {
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
            "dual_clip_ratio": self.dual_clip_ratio,
            "rollout_importance_ratio_cap": self.rollout_importance_ratio_cap,
            "logprob_temperature": self.logprob_temperature,
            "offpolicy_mask_low": self.offpolicy_mask_low,
            "offpolicy_mask_high": self.offpolicy_mask_high,
            "offpolicy_mask_veto_ratio": self.offpolicy_mask_veto_ratio,
        }
        if self.reference_kl_coefficient is not None:
            finite_values["reference_kl_coefficient"] = self.reference_kl_coefficient
        for name, value in finite_values.items():
            _require_finite(name, value)
        if self.clip_low < 0 or self.clip_high < 0:
            raise UnsupportedLearnerConfiguration("policy clipping bounds must be non-negative")
        if self.reference_kl_coefficient is not None and self.reference_kl_coefficient < 0:
            raise UnsupportedLearnerConfiguration("reference KL coefficient must be non-negative")
        if self.policy_loss is PolicyLoss.BEHAVIOR_CLIP and self.use_rollout_importance_sampling:
            raise UnsupportedLearnerConfiguration("behavior_clip cannot be combined with rollout importance sampling")
        if self.use_rollout_importance_sampling and self.rollout_importance_ratio_cap <= 0:
            raise UnsupportedLearnerConfiguration("rollout importance ratio cap must be positive when TIS is enabled")
        if self.update_epochs < 1 or self.max_sequence_length < 1:
            raise UnsupportedLearnerConfiguration("update epochs and max sequence length must be positive")
        if self.logprob_temperature <= 0:
            raise UnsupportedLearnerConfiguration("log-probability temperature must be positive")
        if self.offpolicy_mask_enabled:
            if self.policy_loss is not PolicyLoss.REGULAR:
                raise UnsupportedLearnerConfiguration("the JAX off-policy mask currently requires regular policy loss")
            if self.offpolicy_mask_ratio not in ("mismatch", "full"):
                raise UnsupportedLearnerConfiguration("off-policy mask ratio must be mismatch or full")
            if not (0 < self.offpolicy_mask_veto_ratio <= self.offpolicy_mask_low <= self.offpolicy_mask_high):
                raise UnsupportedLearnerConfiguration(
                    "off-policy mask requires 0 < veto ratio <= low ratio <= high ratio"
                )

    @property
    def requires_behavior_log_probs(self) -> bool:
        """Whether the learner objective or strict transport contract consumes rollout probabilities."""

        return (
            self.require_rollout_logprobs
            or self.offpolicy_mask_enabled
            or self.use_rollout_importance_sampling
            or self.policy_loss is PolicyLoss.BEHAVIOR_CLIP
        )


@dataclass(frozen=True)
class LearnerBatch:
    """One example per row, with response channels aligned on the trailing axis."""

    sequences: np.ndarray
    attention_mask: np.ndarray
    response_mask: np.ndarray
    loss_mask: np.ndarray
    rollout_log_probs: np.ndarray | None
    behavior_policy_versions: np.ndarray

    def __post_init__(self) -> None:
        if self.sequences.ndim != 2 or self.response_mask.ndim != 2:
            raise ValueError("sequences and response_mask must each have two dimensions")
        batch_size, sequence_length = self.sequences.shape
        response_batch_size, response_length = self.response_mask.shape
        if batch_size < 1 or response_length < 1:
            raise ValueError("learner batches and responses must be non-empty")
        if response_batch_size != batch_size or response_length > sequence_length:
            raise ValueError("response channels must align with the sequence rows and trailing positions")
        _require_shape("attention_mask", self.attention_mask, self.sequences.shape)
        _require_shape("loss_mask", self.loss_mask, self.response_mask.shape)
        if self.behavior_policy_versions.shape not in ((batch_size,), self.response_mask.shape):
            raise ValueError(
                "behavior_policy_versions must contain one version per row or one aligned version per response token"
            )
        if self.rollout_log_probs is not None:
            _require_shape("rollout_log_probs", self.rollout_log_probs, self.response_mask.shape)
        for name, mask in (("attention_mask", self.attention_mask), ("response_mask", self.response_mask)):
            if not np.all((mask == 0) | (mask == 1)):
                raise ValueError(f"{name} must contain only zeros and ones")
        if np.any(self.loss_mask < 0) or not np.all(np.isfinite(self.loss_mask)):
            raise ValueError("loss_mask weights must be finite and non-negative")
        if np.any((self.loss_mask > 0) & (self.response_mask == 0)):
            raise ValueError("loss_mask cannot select padded or non-response positions")
        if np.any((self.response_mask > 0) & (self.attention_mask[:, -response_length:] == 0)):
            raise ValueError("response_mask cannot select positions excluded by attention_mask")
        if self.behavior_policy_versions.shape == (batch_size,):
            if np.any(self.behavior_policy_versions < 0):
                raise ValueError("row behavior policy versions must be non-negative")
        elif np.any(self.behavior_policy_versions[self.loss_mask > 0] < 0):
            raise ValueError("every selected response token must have a behavior policy version")
        _require_finite_array("rollout_log_probs", self.rollout_log_probs)

    @property
    def response_length(self) -> int:
        return self.response_mask.shape[1]

    @property
    def selected_behavior_policy_versions(self) -> np.ndarray:
        """Return exact versions for selected tokens, expanding uniform rows lazily."""

        if self.behavior_policy_versions.ndim == 1:
            return np.broadcast_to(self.behavior_policy_versions[:, None], self.response_mask.shape)[self.loss_mask > 0]
        return self.behavior_policy_versions[self.loss_mask > 0]


@dataclass(frozen=True)
class LogProbResult:
    policy_log_probs: np.ndarray
    reference_log_probs: np.ndarray | None
    policy_version: int

    def __post_init__(self) -> None:
        if self.policy_log_probs.ndim != 2:
            raise ValueError("policy_log_probs must have shape [batch, response]")
        if self.reference_log_probs is not None:
            _require_shape("reference_log_probs", self.reference_log_probs, self.policy_log_probs.shape)
        if self.policy_version < 0:
            raise ValueError("policy version must be non-negative")
        _require_finite_array("policy_log_probs", self.policy_log_probs)
        _require_finite_array("reference_log_probs", self.reference_log_probs)


@dataclass(frozen=True)
class UpdateRequest:
    batch: LearnerBatch
    advantages: np.ndarray
    old_policy_log_probs: np.ndarray
    old_policy_version: int
    reference_log_probs: np.ndarray | None
    global_step: int
    global_loss_denominator: float | None

    def __post_init__(self) -> None:
        shape = self.batch.response_mask.shape
        _require_shape("advantages", self.advantages, shape)
        _require_shape("old_policy_log_probs", self.old_policy_log_probs, shape)
        if self.reference_log_probs is not None:
            _require_shape("reference_log_probs", self.reference_log_probs, shape)
        for name, value in (
            ("advantages", self.advantages),
            ("old_policy_log_probs", self.old_policy_log_probs),
            ("reference_log_probs", self.reference_log_probs),
        ):
            _require_finite_array(name, value)
        if self.old_policy_version < 0 or self.global_step < 0:
            raise ValueError("policy version and global step must be non-negative")
        if np.any(self.batch.selected_behavior_policy_versions > self.old_policy_version):
            raise ValueError("behavior policy versions cannot be newer than the recomputed old policy")
        if self.global_loss_denominator is not None and (
            not math.isfinite(self.global_loss_denominator) or self.global_loss_denominator <= 0
        ):
            raise ValueError("global loss denominator must be finite and positive")


@dataclass(frozen=True)
class UpdateResult:
    status: UpdateStatus
    metrics: Mapping[str, float]


@dataclass(frozen=True)
class LearnerState:
    lifecycle: LearnerLifecycle
    policy_version: int
    installed_policy_version: int | None
    update_count: int
    publication_status: PublicationStatus

    @property
    def ready_for_rollouts(self) -> bool:
        return (
            self.lifecycle is LearnerLifecycle.READY
            and self.publication_status is PublicationStatus.INSTALLED
            and self.installed_policy_version == self.policy_version
        )


class Learner(Protocol):
    """Backend-owned model, optimizer, sharding, publication, and checkpoint state."""

    @property
    def state(self) -> LearnerState: ...

    def initialize(self, config: LearnerConfig) -> None: ...

    def compute_log_probs(self, batch: LearnerBatch) -> LogProbResult: ...

    def update(self, request: UpdateRequest) -> UpdateResult: ...

    async def publish_policy(self) -> None: ...

    def save_checkpoint(self, path: str) -> None: ...

    def load_checkpoint(self, path: str) -> None: ...

    def export_policy(self, path: str) -> None: ...

    def close(self) -> None: ...
