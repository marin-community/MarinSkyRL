import pickle

import hydra
import pytest
import skyrl_gym
from skyrl_gym.envs.gsm8k.env import GSM8kEnv
from skyrl_gym.envs.registration import register, registry
from transformers import AutoTokenizer
from taskcompendium.importers.skyrl import gym_task
from taskcompendium.models import Source

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.trajectory_runners.types import TrajectoryID
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import build_tiny_policy

RUNTIME_ENV_ID = "runtime_registered_gsm8k"


@pytest.fixture
def runtime_registered_env():
    register(id=RUNTIME_ENV_ID, entry_point="skyrl_gym.envs.gsm8k.env:GSM8kEnv")
    yield
    registry.pop(RUNTIME_ENV_ID, None)


def test_rollout_worker_can_make_an_environment_the_trainer_registered_at_runtime(runtime_registered_env):
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="ppo_base_config")
    spec = pickle.loads(pickle.dumps(TaskRolloutWorkerSpec.from_config(cfg, engines=[])))
    # A worker process starts with only the environments its imports register.
    del registry[RUNTIME_ENV_ID]

    spec.build(AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B"), WorkerShard(index=0, count=1))

    env = skyrl_gym.make(RUNTIME_ENV_ID, env_config={}, extras={"reward_spec": {"ground_truth": "4"}})
    assert isinstance(env, GSM8kEnv)


@pytest.mark.asyncio
async def test_canonical_worker_runs_real_cpu_inference_across_gym_turns(tmp_path):
    model_path = build_tiny_policy(tmp_path / "model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.step_wise_training = True
    cfg.generator.max_turns = 2
    cfg.generator.max_input_length = 256
    cfg.generator.sampling_params.temperature = 0
    cfg.generator.sampling_params.max_generate_length = 16
    cfg.generator.sampling_params.logprobs = 0
    task = gym_task(
        [{"role": "user", "content": "What is two? End with #### 2."}],
        "gsm8k_multi_turn",
        {"reward_spec": {"ground_truth": "2"}},
        {},
        Source(dataset="tiny", revision="1", row="0", importer_revision="1"),
    )
    worker = TaskRolloutWorkerSpec.from_config(cfg, [CPUInferenceEngine(str(model_path), seed=0)]).build(
        tokenizer, WorkerShard(index=0, count=1)
    )
    try:
        batch = await worker.run(
            {
                "prompts": [[{"role": "user", "content": "What is two? End with #### 2."}]],
                "env_classes": ["gsm8k_multi_turn"],
                "env_extras": [{"task_spec": task.model_dump_json()}],
                "trajectory_ids": [TrajectoryID(task.id, 0)],
                "batch_metadata": None,
                "sampling_params": None,
            }
        )
    finally:
        await worker.shutdown()
    expected_response = tokenizer.encode("#### 1<|im_end|>", add_special_tokens=False)
    assert batch["response_ids"] == [expected_response, expected_response]
    first_prefix = batch["prompt_token_ids"][0] + expected_response
    assert batch["prompt_token_ids"][1][: len(first_prefix)] == first_prefix
    assert batch["loss_masks"] == [[1] * len(expected_response)] * 2
    assert all(len(scores) == len(expected_response) for scores in batch["rollout_logprobs"])
    # The two-turn task gives 0.2 / 2 for each correctly formatted, incorrect answer.
    assert batch["unshaped_rewards"] == [0.1, 0.1]
