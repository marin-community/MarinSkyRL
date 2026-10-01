from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput, ConversationType, ground_truth_from_extras
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

        self.ground_truth = ground_truth_from_extras(extras)
        self.reward_method = env_config.get("reward_method", "strict")
        self.stop_reason = None

    def init(self, prompt: ConversationType) -> tuple[ConversationType, Dict[str, Any]]:
        return prompt, {"chat_completion_params": {}}

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
