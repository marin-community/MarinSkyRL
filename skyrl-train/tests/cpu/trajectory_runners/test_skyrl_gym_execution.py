import pickle

import hydra
import pytest
import skyrl_gym
from skyrl_gym.envs.gsm8k.env import GSM8kEnv
from skyrl_gym.envs.registration import register, registry
from transformers import AutoTokenizer

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.trajectory_runners.skyrl_gym_execution import GymRunnerSpec

RUNTIME_ENV_ID = "runtime_registered_gsm8k"


@pytest.fixture
def runtime_registered_env():
    register(id=RUNTIME_ENV_ID, entry_point="skyrl_gym.envs.gsm8k.env:GSM8kEnv")
    yield
    registry.pop(RUNTIME_ENV_ID, None)


def test_rollout_worker_can_make_an_environment_the_trainer_registered_at_runtime(runtime_registered_env):
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="ppo_base_config")
    spec = pickle.loads(pickle.dumps(GymRunnerSpec.from_config(cfg, engines=[])))
    # A worker process starts with only the environments its imports register.
    del registry[RUNTIME_ENV_ID]

    spec.build(AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"), WorkerShard(index=0, count=1))

    env = skyrl_gym.make(RUNTIME_ENV_ID, env_config={}, extras={"reward_spec": {"ground_truth": "4"}})
    assert isinstance(env, GSM8kEnv)
