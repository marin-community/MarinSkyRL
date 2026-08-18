from typing import Any

from omegaconf import DictConfig

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.ifeval import utils


class IFEvalEnv(BaseTextEnv):
    """Environment for IFEval instruction-following constraint-satisfaction tasks.

    Ground truth is one constraint or a list of constraints. The reward is the fraction
    satisfied. A single constraint therefore retains the original binary behavior.
    """

    def __init__(self, env_config: DictConfig, extras: dict[str, Any] | None = None):
        super().__init__()

        extras = extras or {}
        assert "reward_model" in extras, "reward_model field is required"
        assert "ground_truth" in extras["reward_model"], "ground_truth is required in reward_model field"
        self.ground_truth = extras["reward_model"]["ground_truth"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step

        score_info = utils.compute_score(action, self.ground_truth)
        reward = score_info["score"]
        metadata = {key: value for key, value in score_info.items() if key != "score"}

        # No observation in ifeval, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata=metadata)
