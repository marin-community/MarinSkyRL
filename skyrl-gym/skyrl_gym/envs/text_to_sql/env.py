"""Single-turn text-to-SQL execution environment.

Malformed ground truth is logged and scores zero with verifier-error metadata
rather than crashing a distributed rollout worker; the dataset-preparation
contract (``skyrl_gym.envs.data_contracts``) rejects those rows up front.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.text_to_sql.scoring import parse_ground_truth, score

logger = logging.getLogger(__name__)
_INVALID_GROUND_TRUTH_ERROR = "invalid reward_model.ground_truth"


class TextToSQLEnv(BaseTextEnv):
    """Result-set-equivalence text-to-SQL verifier. One step, then done."""

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()
        self.verifyit_enabled = bool(env_config.get("verifyit_enabled", False))
        reward_model = (extras or {}).get("reward_model")
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
        self._ground_truth = ground_truth if self.verifyit_enabled or isinstance(ground_truth, str) else None
        if not self.verifyit_enabled and parse_ground_truth(self._ground_truth) is None:
            logger.error("text_to_sql: %s=%r; scoring 0.", _INVALID_GROUND_TRUTH_ERROR, ground_truth)
            self._ground_truth = None

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.verifyit_enabled:
            from skyrl_gym.envs.sqlite_verifyit import grade_sql, project_sql_reward
            from skyrl_gym.verification import VerificationResult
            from verifyit.grade import Status

            result = grade_sql(self._ground_truth, action)
            diagnostics = result.diagnostics()
            verification = (
                VerificationResult.verified(result.verdict.reward, diagnostics=diagnostics)
                if result.verdict.status is Status.SCORED
                else VerificationResult.error("SQL verification failed", diagnostics=diagnostics)
            )
            return BaseTextEnvStepOutput(
                observations=[],
                reward=project_sql_reward(result),
                done=True,
                metadata=diagnostics,
                verification=verification,
            )
        if self._ground_truth is None:
            result = BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"verifier_error": _INVALID_GROUND_TRUTH_ERROR},
            )
            return result
        try:
            reward, metadata = score(self._ground_truth, action)
        except RuntimeError:
            from skyrl_gym.verification import VerificationResult

            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={},
                verification=VerificationResult.error(
                    "SQL verification failed",
                    diagnostics={},
                ),
            )
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata=metadata)
