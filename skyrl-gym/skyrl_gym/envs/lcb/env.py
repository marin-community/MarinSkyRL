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
        super().__init__(env_config)
        self.sandbox_config = env_config.get("sandbox", {})
        self.code_verifier_config = env_config.get("code_verifier", {})
        self.reward_mode = str(env_config.get("reward_mode", BINARY_REWARD_MODE))
        if self.reward_mode not in LCB_REWARD_MODES:
            raise ValueError(f"Unsupported LCB reward_mode: {self.reward_mode!r}.")

        reward_model = (extras or {}).get("reward_model")
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
        if self.verifyit_enabled:
            self.raw_ground_truth = ground_truth
            self.tests = None
            return
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
        if self.tests is None and not self.verifyit_enabled:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"parsed_code": None, "verifier_error": _INVALID_GROUND_TRUTH_ERROR},
            )
        if self.verifyit_enabled:
            from skyrl_gym.verification import VerificationResult

            preparation_errors = ()
            invalid_task_errors = ()
            try:
                from harbor_config.errors import error_category
                from verifyit.grade import InvalidTask
                from verifyit.preparation.errors import PreparationError
                from skyrl_gym.envs.lcb.verifyit_execution import CodePolicy, execute_code_verifyit
                from skyrl_gym.envs.lcb.livecodebench import DEFAULT_LIMITS, VerifierLimits
                from skyrl_gym.envs.nemotron_ultra.sandbox import SandboxClient

                preparation_errors = (PreparationError,)
                invalid_task_errors = (InvalidTask,)
                reward, detail = execute_code_verifyit(
                    self.raw_ground_truth,
                    action,
                    policy=CodePolicy.LCB,
                    fractional=self.reward_mode == "fractional",
                    timeout=int(self.code_verifier_config.get("per_test_timeout_seconds", 6)),
                    limits=VerifierLimits(
                        max_memory_bytes=self.code_verifier_config.get(
                            "max_memory_bytes", DEFAULT_LIMITS.max_memory_bytes
                        ),
                        total_timeout_seconds=self.code_verifier_config.get(
                            "total_timeout_seconds", DEFAULT_LIMITS.total_timeout_seconds
                        ),
                    ),
                    sandbox=SandboxClient(
                        host=str(self.sandbox_config.get("host", "127.0.0.1")),
                        port=int(self.sandbox_config.get("port", 6000)),
                    ),
                )
                parsed_code = detail["parsed_code"]
                detail["comparisons"] = [
                    {k: v for k, v in comparison.items() if k in {"module", "candidate", "reward"}}
                    for comparison in detail.get("comparisons", [])
                ]
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=reward,
                    done=True,
                    metadata={"parsed_code": parsed_code, **detail},
                    verification=VerificationResult.verified(reward, passed=reward == 1, diagnostics=detail),
                )
            except Exception as error:
                logger.exception("LCB verification boundary failed")
                detail = {
                    "verifyit_status": "infra_error",
                    "error_category": "infrastructure",
                    "error_type": type(error).__name__,
                    "error_message": "Code verification unavailable",
                    "preparation_stage": "code_boundary",
                }
                if isinstance(error, invalid_task_errors):
                    detail.update(verifyit_status="invalid_task", error_category=error_category("InvalidTask").value)
                if isinstance(error, preparation_errors):
                    detail.update(error.verdict.detail)
                    detail.update(
                        verifyit_status=error.verdict.status.value,
                        error_category=error.failure.category.value,
                        preparation_stage=error.failure.stage,
                    )
                detail.setdefault(
                    "preparation",
                    {
                        "policy": "lcb_source_v1",
                        "raw_response": action,
                    },
                )
                return BaseTextEnvStepOutput(
                    observations=[],
                    reward=0.0,
                    done=True,
                    metadata=detail,
                    verification=VerificationResult.error("Code verification unavailable", diagnostics=detail),
                )
        else:
            parsed_code, reward = compute_score(action, self.tests, self.reward_mode)

        # RL on LCB w/ single-turn
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"parsed_code": parsed_code})
