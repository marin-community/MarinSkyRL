"""Deterministic CPU fake for tests of the learner orchestration contract."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

import numpy as np

from skyrl_train.learner import (
    LearnerBatch,
    LearnerConfig,
    LearnerLifecycle,
    LearnerState,
    LogProbResult,
    PolicyLoss,
    PublicationStatus,
    UpdateRequest,
    UpdateResult,
    UpdateStatus,
)


_CHECKPOINT_FILENAME = "fake_learner_state.json"
_REFERENCE_PARAMETER = -0.5


class FakeLearnerOperation(StrEnum):
    LOG_PROBS = "log_probs"
    UPDATE = "update"
    PUBLISH = "publish"
    SAVE = "save"
    LOAD = "load"
    CLOSE = "close"


class FakeLearnerError(RuntimeError):
    """A failure requested by a test through ``fail_next``."""


class StatefulFakeLearner:
    """Stateful learner fake; production configuration must never select this class."""

    def __init__(self, *, initial_parameter: float = 0.25):
        self._parameter = initial_parameter
        self._policy_version = 0
        self._installed_policy_version: int | None = None
        self._update_count = 0
        self._publication_status = PublicationStatus.NOT_STARTED
        self._lifecycle = LearnerLifecycle.UNINITIALIZED
        self._config: LearnerConfig | None = None
        self._failures: set[FakeLearnerOperation] = set()
        self._defer_publication = False

    @property
    def state(self) -> LearnerState:
        return LearnerState(
            lifecycle=self._lifecycle,
            policy_version=self._policy_version,
            installed_policy_version=self._installed_policy_version,
            update_count=self._update_count,
            publication_status=self._publication_status,
        )

    def initialize(self, config: LearnerConfig) -> None:
        if self._lifecycle is LearnerLifecycle.CLOSED:
            raise RuntimeError("cannot initialize a closed learner")
        if self._config is not None and self._config != config:
            raise RuntimeError("learner was already initialized with a different configuration")
        self._config = config
        self._lifecycle = LearnerLifecycle.READY

    def fail_next(self, operation: FakeLearnerOperation) -> None:
        self._failures.add(operation)

    def defer_next_publication(self) -> None:
        self._defer_publication = True

    def complete_pending_publication(self) -> None:
        self._require_ready()
        if self._publication_status is not PublicationStatus.PENDING:
            raise RuntimeError("no fake policy publication is pending")
        self._installed_policy_version = self._policy_version
        self._publication_status = PublicationStatus.INSTALLED

    def compute_log_probs(self, batch: LearnerBatch) -> LogProbResult:
        config = self._require_ready()
        self._maybe_fail(FakeLearnerOperation.LOG_PROBS)
        return LogProbResult(
            policy_log_probs=self._log_probs(batch, self._parameter),
            reference_log_probs=(
                self._log_probs(batch, _REFERENCE_PARAMETER) if config.requires_reference_log_probs else None
            ),
            policy_version=self._policy_version,
        )

    def update(self, request: UpdateRequest) -> UpdateResult:
        config = self._require_ready()
        self._maybe_fail(FakeLearnerOperation.UPDATE)
        if request.old_policy_version != self._policy_version:
            raise ValueError("old policy log probabilities name a different learner version")
        if not np.allclose(
            request.old_policy_log_probs,
            self._log_probs(request.batch, self._parameter),
            rtol=0,
            atol=1e-6,
        ):
            raise ValueError("old policy log probabilities do not match the named learner version")
        if config.requires_reference_log_probs:
            if request.reference_log_probs is None or not np.allclose(
                request.reference_log_probs,
                self._log_probs(request.batch, _REFERENCE_PARAMETER),
                rtol=0,
                atol=1e-6,
            ):
                raise ValueError("reference log probabilities do not match the fake reference policy")

        uses_behavior_probs = config.use_rollout_importance_sampling or config.policy_loss is PolicyLoss.BEHAVIOR_CLIP
        if uses_behavior_probs and request.batch.rollout_log_probs is None:
            raise ValueError("the configured update requires rollout log probabilities")

        weights = request.batch.loss_mask.astype(np.float64, copy=False)
        denominator = float(weights.sum())
        if denominator == 0:
            return UpdateResult(status=UpdateStatus.SKIPPED, metrics=self._metrics(request, denominator, 0.0))

        response_tokens = request.batch.sequences[:, -request.batch.response_length :]
        terms = request.advantages * (1.0 + np.remainder(response_tokens, 7) / 10.0)
        if uses_behavior_probs:
            assert request.batch.rollout_log_probs is not None
            ratios = np.exp(np.clip(request.old_policy_log_probs - request.batch.rollout_log_probs, -8.0, 8.0))
            if config.use_rollout_importance_sampling:
                ratios = np.minimum(ratios, config.rollout_importance_ratio_cap)
            terms = terms * ratios
        update_signal = float(np.sum(terms * weights) / denominator)

        self._parameter += update_signal * 0.01
        self._update_count += 1
        self._policy_version += 1
        self._publication_status = PublicationStatus.OUTDATED
        return UpdateResult(
            status=UpdateStatus.SUCCEEDED,
            metrics=self._metrics(request, denominator, update_signal),
        )

    async def publish_policy(self) -> None:
        self._require_ready()
        if FakeLearnerOperation.PUBLISH in self._failures:
            self._failures.remove(FakeLearnerOperation.PUBLISH)
            self._publication_status = PublicationStatus.FAILED
            raise FakeLearnerError("requested fake learner failure during publish")
        if self._defer_publication:
            self._defer_publication = False
            self._publication_status = PublicationStatus.PENDING
            return
        self._installed_policy_version = self._policy_version
        self._publication_status = PublicationStatus.INSTALLED

    def save_checkpoint(self, path: str) -> None:
        self._require_ready()
        self._maybe_fail(FakeLearnerOperation.SAVE)
        checkpoint_dir = Path(path)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "parameter": self._parameter,
            "policy_version": self._policy_version,
            "update_count": self._update_count,
        }
        (checkpoint_dir / _CHECKPOINT_FILENAME).write_text(json.dumps(payload, sort_keys=True))

    def load_checkpoint(self, path: str) -> None:
        self._require_ready()
        self._maybe_fail(FakeLearnerOperation.LOAD)
        state_path = Path(path) / _CHECKPOINT_FILENAME
        payload = json.loads(state_path.read_text())
        self._parameter = float(payload["parameter"])
        self._policy_version = int(payload["policy_version"])
        self._update_count = int(payload["update_count"])
        self._installed_policy_version = None
        self._publication_status = PublicationStatus.OUTDATED

    def export_policy(self, path: str) -> None:
        self._require_ready()
        export_dir = Path(path)
        export_dir.mkdir(parents=True, exist_ok=True)
        (export_dir / "fake_policy.json").write_text(json.dumps({"parameter": self._parameter}, sort_keys=True))

    def close(self) -> None:
        self._maybe_fail(FakeLearnerOperation.CLOSE)
        self._lifecycle = LearnerLifecycle.CLOSED

    def _log_probs(self, batch: LearnerBatch, parameter: float) -> np.ndarray:
        tokens = batch.sequences[:, -batch.response_length :].astype(np.float64, copy=False)
        positions = np.arange(batch.response_length, dtype=np.float64)[None, :]
        values = -np.log1p(np.remainder(tokens, 31) + 1.0) - positions * 0.001 - parameter * 0.01
        return np.where(batch.response_mask > 0, values, 0.0).astype(np.float32)

    def _metrics(self, request: UpdateRequest, masked_tokens: float, update_signal: float) -> dict[str, float]:
        return {
            "fake/masked_tokens": masked_tokens,
            "fake/update_signal": update_signal,
            "fake/global_loss_denominator": float(request.global_loss_denominator or -1.0),
            "fake/oldest_behavior_version": float(request.batch.behavior_policy_versions.min()),
            "fake/newest_behavior_version": float(request.batch.behavior_policy_versions.max()),
        }

    def _require_ready(self) -> LearnerConfig:
        if self._lifecycle is not LearnerLifecycle.READY or self._config is None:
            raise RuntimeError(f"fake learner is not ready: {self._lifecycle}")
        return self._config

    def _maybe_fail(self, operation: FakeLearnerOperation) -> None:
        if operation in self._failures:
            self._failures.remove(operation)
            raise FakeLearnerError(f"requested fake learner failure during {operation}")
