import json
import logging
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.lcb.livecodebench import compute_score

logger = logging.getLogger(__name__)


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
        json_error = False
        try:
            tests = json.loads(ground_truth) if isinstance(ground_truth, str) else None
        except json.JSONDecodeError:
            logger.exception("lcb: invalid reward_model.ground_truth=%r; scoring 0.", ground_truth)
            json_error = True
            tests = None
        self.tests = tests if isinstance(tests, list) and tests else None
        if self.tests is None and not json_error:
            logger.error("lcb: invalid reward_model.ground_truth=%r; scoring 0.", ground_truth)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        if self.tests is None:
            return BaseTextEnvStepOutput(
                observations=[],
                reward=0.0,
                done=True,
                metadata={"parsed_code": None, "verifier_error": "invalid reward_model.ground_truth"},
            )
        parsed_code, reward = compute_score(action, self.tests)

        # RL on LCB w/ single-turn
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=True, metadata={"parsed_code": parsed_code})
