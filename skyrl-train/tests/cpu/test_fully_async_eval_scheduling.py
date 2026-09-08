"""Run the actual driver/evaluator through installed and background schedules."""

import asyncio

import pytest

from tests.cpu.test_fully_async_publication_cadence import Runner, make_driver
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_fully_async_cfg


@pytest.mark.asyncio
async def test_installed_weights_eval_does_not_publish_off_grid():
    trainer = make_driver(eval_steps=3)
    trainer.cfg.trainer.fully_async.eval_on_installed_weights = True
    await asyncio.wait_for(trainer._train_loop(), 10)
    assert trainer.inference_engine_client.publications == [0, 2, 4, 5]
    assert trainer.trajectory_runner.evaluations == [(0, 0), (3, 2), (5, 5)]
    metrics = next(m for step, m in trainer.tracker.rows if step == 3 and "trainer/global_step" in m)
    assert metrics["eval/policy_version_lag"] == 1
    assert "timing/eval_weight_sync" not in metrics


class PausableRunner(Runner):
    async def run(self, request, **kwargs):
        # Native single-prompt retry waits for resume; unlike the legacy fake's
        # batched-path rejection, this boundary models the stock gym runner path.
        await self.engine.ready.wait()
        return await super().run(request, **kwargs)


def background_driver(**kwargs):
    trainer = make_driver(runner_type=PausableRunner, **kwargs)
    trainer.cfg.trainer.fully_async.eval_on_installed_weights = True
    trainer.cfg.trainer.fully_async.eval_mode = "background"
    # The service fake is the only substituted I/O surface. Production rejects
    # this non-stock runner; the guard has its own negative test below.
    trainer._validate_background_eval_runner = lambda: None
    return trainer


@pytest.mark.asyncio
async def test_background_eval_does_not_block_next_step_and_freezes_dump_identity():
    trainer = background_driver(eval_steps=3)
    release = asyncio.Event()
    entered = asyncio.Event()
    original_run = trainer.trajectory_runner.run
    original_train = trainer._run_training

    async def delayed_eval(request, **kwargs):
        metadata = request["batch_metadata"]
        if metadata.training_phase == "eval" and metadata.global_step == 3:
            entered.set()
            await release.wait()
        return await original_run(request, **kwargs)

    async def training_step(batch):
        result = await original_train(batch)
        if trainer.policy_model.completed_update == 4:
            await asyncio.wait_for(entered.wait(), 2)
            assert [step for step, _ in trainer.policy_model.consumed] == [1, 2, 3, 4]
            assert not any("eval/requested_at_step" in m for _, m in trainer.tracker.rows)
            release.set()
        return result

    trainer.trajectory_runner.run = delayed_eval
    trainer._run_training = training_step
    await asyncio.wait_for(trainer._train_loop(), 10)
    rows = [(step, m) for step, m in trainer.tracker.rows if m.get("eval/requested_at_step") == 3]
    assert rows and all(step >= 4 for step, _ in rows)
    assert rows[0][1]["eval/policy_version_requested"] == 2
    assert rows[0][1]["eval/policy_version_completed"] >= 4
    assert [step for step, _ in trainer.trajectory_runner.evaluations] == [0, 3, 5]
    assert trainer._last_successful_eval_step == 5
    assert trainer.inference_engine_client.publications == [0, 2, 4, 5]
    assert trainer._background_eval_tasks == []


@pytest.mark.asyncio
async def test_background_eval_is_awaited_before_final_sync_and_deduped():
    trainer = background_driver(eval_steps=5)
    await asyncio.wait_for(trainer._train_loop(), 10)
    assert trainer.trajectory_runner.evaluations == [(0, 0), (5, 5)]
    assert trainer._last_successful_eval_step == 5
    assert trainer._background_eval_tasks == []
    assert any(m.get("eval/requested_at_step") == 5 for _, m in trainer.tracker.rows)


@pytest.mark.asyncio
async def test_weight_sync_refuses_a_background_task():
    trainer = make_driver()
    trainer._weight_sync_owner = asyncio.current_task()
    with pytest.raises(RuntimeError, match="training driver task"):
        await asyncio.create_task(trainer._publish_policy_weights(reason="evaluation", timing_name="eval_weight_sync"))
    assert trainer.inference_engine_client.publications == []


def test_background_eval_rejects_unverified_runner():
    trainer = make_driver()
    with pytest.raises(ValueError, match="stock whole-trajectory"):
        trainer._validate_background_eval_runner()


@pytest.mark.parametrize("installed,mode", [(False, "background"), (True, "other"), (1, "blocking")])
def test_invalid_background_configuration_rejects(installed, mode):
    cfg = get_default_config()
    cfg.trainer.fully_async.eval_on_installed_weights = installed
    cfg.trainer.fully_async.eval_mode = mode
    with pytest.raises(ValueError, match="evaluation|eval_"):
        validate_fully_async_cfg(cfg)


@pytest.mark.asyncio
async def test_stock_runner_interleaves_train_and_eval_with_same_locked_sink(monkeypatch):
    from unittest.mock import MagicMock
    from types import SimpleNamespace
    from tests.cpu.trajectory_runners.test_skyrl_gym_runner import mock_tokenizer as tokenizer_fixture
    from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner
    from skyrl_train.trajectory_runners.trajectory_retention import make_trajectory_sink
    from skyrl_train.trajectory_runners.trajectory_processing import prepare_trajectory_request
    from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
    from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer

    tokenizer = tokenizer_fixture.__wrapped__()
    cfg = get_default_config()
    cfg.generator.batched = False
    cfg.generator.use_conversation_multi_turn = False
    cfg.generator.chat_template = {"source": "name", "name_or_path": None}
    cfg.generator.chat_template_kwargs = {}
    cfg.generator.sampling_params.logprobs = None
    cfg.generator.max_input_length = 512
    cfg.generator.max_turns = 1
    environment_cfg = MagicMock()
    environment_cfg.max_env_workers = 0
    both_entered = asyncio.Event()
    submitted = []

    async def generate(request):
        submitted.append(request)
        if len(submitted) == 2:
            both_entered.set()
        await both_entered.wait()
        return {"responses": ["four"], "response_ids": [[1, 4]], "stop_reasons": ["stop"]}

    environments = []

    def make_environment(*args, **kwargs):
        env = MagicMock()
        env.init.return_value = ([{"role": "user", "content": "2+2?"}], {})
        env.step.return_value = BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
        environments.append(env)
        return env

    monkeypatch.setattr("skyrl_gym.make", make_environment)
    runner = SkyRLGymTrajectoryRunner(cfg.generator, environment_cfg, SimpleNamespace(generate=generate), tokenizer)
    sink = make_trajectory_sink(cfg.generator, tokenizer)
    runner.set_trajectory_sink(sink)
    owner = SimpleNamespace(trajectory_runner=runner, trajectory_sink=sink, cfg=cfg)
    FullyAsyncRayPPOTrainer._validate_background_eval_runner(owner)
    retained = []
    original_retain = sink.retain

    def record_retain(request, output):
        retained.append(request["batch_metadata"].training_phase)
        return original_retain(request, output)

    monkeypatch.setattr(sink, "retain", record_retain)
    requests = []
    for phase in ("train", "eval"):
        request, _ = prepare_trajectory_request(
            [{"uid": phase, "prompt": [{"role": "user", "content": "2+2?"}], "env_class": "gsm8k", "env_extras": {}}],
            1,
            {},
            "gsm8k",
            phase,
            3,
        )
        requests.append(request)
    runner.set_trajectory_sink(sink)
    await runner.start_eval_session(run_name="test", eval_step=3)
    outputs = await asyncio.wait_for(asyncio.gather(*(runner.run(request) for request in requests)), 3)
    await runner.stop_eval_session()
    assert both_entered.is_set() and len(environments) == 2
    assert sorted(retained) == ["eval", "train"]
    assert [output["trajectory_ids"][0].instance_id for output in outputs] == ["train", "eval"]
    assert runner.trajectory_sink is sink
    sink.close()


@pytest.mark.asyncio
async def test_base_eval_captures_success_identity_before_await(monkeypatch):
    from skyrl_train.trainer import RayPPOTrainer

    trainer = make_driver()
    trainer.global_step = 3
    observed = []

    async def evaluator(**kwargs):
        observed.append(kwargs["global_step"])
        await asyncio.sleep(0)
        trainer.global_step = 7
        return {"eval/all/avg_score": 0.5}

    monkeypatch.setattr("skyrl_train.trainer.evaluate", evaluator)
    await RayPPOTrainer.eval(trainer)
    assert observed == [3]
    assert trainer._last_successful_eval_step == 3
