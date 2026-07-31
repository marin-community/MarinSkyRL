import json
import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.data_contracts import normalize_lcb_ground_truth
from skyrl_gym.envs.lcb.livecodebench import compute_score

logger = logging.getLogger(__name__)
_INVALID_GROUND_TRUTH_ERROR = "invalid reward_model.ground_truth"


class LCBEnv(BaseTextEnv):
    """
    Environment for LiveCodeBench execution environment.
    """

    def __init__(
        self,
        env_config: DictConfig,
        extras: dict[str, Any] | None = None,
    ):
        super().__init__()

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
        parsed_code, reward = compute_score(action, self.tests)

        # RL on LCB w/ single-turn
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"parsed_code": parsed_code})
