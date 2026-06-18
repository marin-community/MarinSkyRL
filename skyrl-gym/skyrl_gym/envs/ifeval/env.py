from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.ifeval import utils
from typing import Dict, Any
from omegaconf import DictConfig


class IFEvalEnv(BaseTextEnv):
    """Environment for IFEval instruction-following constraint-satisfaction tasks.

    Scores the model's response against the IFEval constraint named in
    ``extras["reward_model"]["ground_truth"]`` (a JSON spec with ``func_name`` + kwargs),
    rather than a boxed-math answer match. Reward is 1.0 if the constraint is satisfied,
    0.0 otherwise. Mirrors :class:`AIMEEnv` (single step, no tool calls, no observation).
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_model" in extras, "reward_model field is required"
        assert "ground_truth" in extras["reward_model"], (
            "ground_truth is required in reward_model field"
        )
        self.ground_truth = extras["reward_model"]["ground_truth"]

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step

        score_info = utils.compute_score(action, self.ground_truth)
        reward = score_info["score"]
        metadata = {"acc": score_info["acc"], "func_name": score_info["func_name"]}

        # No observation in ifeval, and no tool call
        return BaseTextEnvStepOutput(
            observations=[], reward=reward, done=done, metadata=metadata
        )
