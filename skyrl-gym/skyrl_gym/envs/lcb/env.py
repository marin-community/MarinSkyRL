import json
import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.lcb.livecodebench import (
    BINARY_REWARD_MODE,
    LCB_REWARD_MODES,
    compute_score,
    normalize_lcb_ground_truth,
)

logger = logging.getLogger(__name__)
_INVALID_GROUND_TRUTH_ERROR = "invalid reward_model.ground_truth"


class LCBEnv(BaseTextEnv):
    """LiveCodeBench execution environment.

    Malformed ground truth is logged and scores zero with verifier-error metadata
    rather than crashing a distributed rollout worker.
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: dict[str, Any] | None = None,
    ):
        super().__init__()
        self.sandbox_config = env_config.get("sandbox", {})
        self.verifyit_enabled = bool(env_config.get("verifyit_enabled", False))
        self.reward_mode = str(env_config.get("reward_mode", BINARY_REWARD_MODE))
        if self.reward_mode not in LCB_REWARD_MODES:
            raise ValueError(f"Unsupported LCB reward_mode: {self.reward_mode!r}.")

        reward_model = (extras or {}).get("reward_model")
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
        try:
            normalized = normalize_lcb_ground_truth(ground_truth) if isinstance(ground_truth, str) else None
            tests = json.loads(normalized) if normalized is not None else None
        except (TypeError, ValueError):
            logger.exception("lcb: %s=%r; scoring 0.", _INVALID_GROUND_TRUTH_ERROR, ground_truth)
            tests = None
        self.tests = tests if isinstance(tests, list) and tests else None
        if self.tests is None and not isinstance(ground_truth, str):
            logger.error("lcb: %s=%r; scoring 0.", _INVALID_GROUND_TRUTH_ERROR, ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.tests is None:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"parsed_code": None, "verifier_error": _INVALID_GROUND_TRUTH_ERROR},
            )
        if self.verifyit_enabled:
            from skyrl_gym.envs.lcb.livecodebench import extract_code_from_model
            from skyrl_gym.envs.lcb.verifyit_execution import execute_code_verifyit
            from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient
            from skyrl_gym.verification import VerificationResult

            parsed_code = extract_code_from_model(action)
            try:
                reward = execute_code_verifyit(
                    self.tests,
                    parsed_code or "",
                    fractional=self.reward_mode == "fractional",
                    sandbox=SandboxClient(
                        host=str(self.sandbox_config.get("host", "127.0.0.1")),
                        port=int(self.sandbox_config.get("port", 6000)),
                    ),
                )[0]
            except (ImportError, RuntimeError, ValueError, TypeError, OSError):
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata={},
                    verification=VerificationResult.error("Code verification unavailable"),
                )
        else:
            parsed_code, reward = compute_score(action, self.tests, self.reward_mode)

        # RL on LCB w/ single-turn
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"parsed_code": parsed_code})
