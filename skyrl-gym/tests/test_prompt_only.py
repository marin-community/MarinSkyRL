from omegaconf import DictConfig

from skyrl_gym.envs.prompt_only.env import PromptOnlyEnv


def test_prompt_only_returns_terminal_zero_reward_for_any_completion():
    env = PromptOnlyEnv(DictConfig({}), extras={"data_source": "example"})

    output = env.step("A model-generated answer")

    assert output == {"observations": [], "reward": 0.0, "done": True, "metadata": {}}
