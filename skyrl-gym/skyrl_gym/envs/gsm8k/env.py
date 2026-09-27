from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.envs.gsm8k import utils
from typing import Dict, Any
from omegaconf import DictConfig
from skyrl_gym.verification import RolloutEvidence


class GSM8kEnv(BaseTextEnv):
    """
    Environment for Math execution tasks.
    """

    def __init__(self, env_config: DictConfig, extras: Dict[str, Any] = {}):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], "ground_truth is required in reward_spec field"
        self.ground_truth = extras["reward_spec"]["ground_truth"]
        self.answer_format = extras["reward_spec"].get("answer_format", "strict")
        self.stop_reason = None

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.stop_reason = evidence.stop_reason

    def _get_reward(self, action: str) -> float:
        if self.answer_format == "final_line" and self.stop_reason not in {"stop", "complete", "eos", "end_turn"}:
            return 0.0
        return utils.compute_score(action, self.ground_truth, method=self.answer_format)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
