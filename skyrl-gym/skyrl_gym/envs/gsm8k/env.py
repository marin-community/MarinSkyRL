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

        reward_spec = extras.get("reward_spec") or extras.get("reward_model")
        assert reward_spec is not None, "reward_spec (or reward_model) field is required"
        assert "ground_truth" in reward_spec, "ground_truth is required in reward_spec field"
        self.ground_truth = reward_spec["ground_truth"]
        self.reward_method = env_config.get("reward_method", "strict")
        self.stop_reason = None

    def set_rollout_evidence(self, evidence: RolloutEvidence) -> None:
        self.stop_reason = evidence.stop_reason

    def _get_reward(self, action: str) -> float:
        if self.reward_method == "final_line" and self.stop_reason not in {"stop", "complete", "eos", "end_turn"}:
            return 0.0
        return utils.compute_score(action, self.ground_truth, method=self.reward_method)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        done = True  # always done after one step
        reward = self._get_reward(action)
        # No observation in gsm8k, and no tool call
        return BaseTextEnvStepOutput(observations=[], reward=reward, done=done, metadata={})
