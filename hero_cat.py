"""Task-owned repeat-cat prompt and exact-output rule."""

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


RESPONSE_LIMIT = 128
ACCEPTED_STOPS = {"stop", "complete", "eos", "end_turn"}
COMPLETION_TEMPLATE = "{{ bos_token }}{{ messages[0]['content'] }}"


def cat_prompt(count, verb="Repeat"):
    return f"{verb} cat 2 times: cat cat\n{verb} cat {count} times:"


def cat_reward(response, count, stop_reason):
    # Outer whitespace includes the terminating newline. Internal spacing,
    # spelling and case must match; length-capped responses receive no reward.
    return float(stop_reason in ACCEPTED_STOPS and response.strip() == " ".join(["cat"] * count))


class CatRepeatEnv(BaseTextEnv):
    def __init__(self, env_config, extras):
        super().__init__()
        self.count = int(extras["reward_spec"]["count"])
        self.stop_reason = None

    def set_rollout_evidence(self, evidence):
        self.stop_reason = evidence.stop_reason

    def step(self, action: str) -> BaseTextEnvStepOutput:
        return BaseTextEnvStepOutput(
            observations=[], reward=cat_reward(action, self.count, self.stop_reason), done=True, metadata={}
        )
