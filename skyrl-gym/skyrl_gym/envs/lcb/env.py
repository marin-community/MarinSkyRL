import json
from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.lcb.livecodebench import compute_score


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
            tests = json.loads(ground_truth) if isinstance(ground_truth, str) else None
        except json.JSONDecodeError:
            tests = None
        self.tests = tests if isinstance(tests, list) and tests else None

    def _get_reward(self, action: str) -> float:
        if self.tests is None:
            return 0.0
        _, reward = compute_score(action, self.tests)
        return reward

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
