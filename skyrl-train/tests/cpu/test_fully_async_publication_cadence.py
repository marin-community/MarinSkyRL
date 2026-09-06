"""Exercise the async driver with in-memory learner and inference services."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

from skyrl_train.callbacks import TrainerCallback
from skyrl_train.callbacks.builtin import EvaluationCallback
from skyrl_train.config.utils import CONFIG_DIR, DEFAULT_CONFIG_NAME, get_default_config
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.trainer_utils import ResumeMode
from skyrl_train.utils.utils import validate_cfg


class PromptRows:
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        return {
            "uid": str(index),
            "prompt": [{"role": "user", "content": "1+1?"}],
            "env_class": "gsm8k",
            "env_extras": {},
        }

    @staticmethod
    def collate_fn(rows):
        return rows


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def decode(self, tokens):
        return str(list(tokens))


class Tracker:
    def __init__(self):
        self.rows = []

    def log(self, metrics, *, step, commit=False):
        self.rows.append((step, dict(metrics)))


class InferenceService:
    def __init__(self):
        self.installed_update = None
        self.publications = []
        self.ready = asyncio.Event()
        self.ready.set()

    async def pause_generation(self):
        self.ready.clear()

    async def resume_generation(self):
        self.ready.set()


class LearnerService:
    def __init__(self, completed_update=0, fail_update=None):
        self.completed_update = completed_update
        self.fail_update = fail_update
        self.actor_infos = [SimpleNamespace(rank=SimpleNamespace(dp_size=1))]
        self.consumed = []

    async def async_run_method(self, _dispatch, method, engine):
        assert method == "broadcast_to_inference_engines"
        if self.completed_update == self.fail_update:
            raise RuntimeError("publication transport failed")
        engine.installed_update = self.completed_update
        engine.publications.append(self.completed_update)

    def async_run_ray_method(self, _dispatch, method):
        assert method == "barrier_all"

        async def remote_barrier():
            # Remote barriers yield to producers while weight publication is paused.
            completed = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_soon(completed.set_result, None)
            await completed

        return [remote_barrier()]


class Runner:
    """Generate tokens encoding the installed update, with optional runner stamps."""

    def __init__(self, engine, *, report_step):
        self.engine = engine
        self.report_step = report_step
        self.global_step_fn = None
        self.evaluations = []
        self.generations = []

    def set_trajectory_sink(self, sink):
        self.sink = sink

    async def start_eval_session(self, **kwargs):
        pass

    async def stop_eval_session(self):
        pass

    async def run(self, request, **kwargs):
        # The direct inference client rejects a new generate() call while paused;
        # it only waits for resume after a request has already entered its retry loop.
        if not self.engine.ready.is_set():
            raise RuntimeError("pause_generation is unsupported for InferenceEngineClient.generate().")
        version = self.engine.installed_update
        assert version is not None
        ids = request["trajectory_ids"]
        if request["batch_metadata"].training_phase == "eval":
            self.evaluations.append((request["batch_metadata"].global_step, version))
        else:
            self.generations.append((ids[0].instance_id, version))
        batch = {
            "prompt_token_ids": [[1] for _ in ids],
            "response_ids": [[10 + version] for _ in ids],
            "rewards": [float(index % 2) for index, _ in enumerate(ids)],
            "unshaped_rewards": [float(index % 2) for index, _ in enumerate(ids)],
            "loss_masks": [[1] for _ in ids],
            "stop_reasons": ["stop" for _ in ids],
            "rollout_metrics": {},
            "rollout_logprobs": [[-0.5] for _ in ids],
            "trajectory_ids": ids,
            "is_last_step": [True for _ in ids],
            "exclude_from_baseline": [False for _ in ids],
        }
        if self.report_step and self.global_step_fn is not None:
            batch["actual_global_step"] = self.global_step_fn()
        return batch


class LogAndStop(TrainerCallback):
    def __init__(self, stop_step=None, save_step=None):
        self.stop_step = stop_step
        self.save_step = save_step

    def on_step_end(self, state, control, **kwargs):
        control.should_log = True
        control.should_save = state.global_step == self.save_step
        control.should_training_stop = state.global_step == self.stop_step
        return control


class DriverWithCpuLearner(FullyAsyncRayPPOTrainer):
    """Replace remote model initialization/update with a scalar learner service.

    The actual driver, producers, dataloader, admission, batch conversion,
    weight publication, callbacks and evaluation all run unchanged.
    """

    def init_weight_sync_state(self):
        pass

    def save_checkpoints(self):
        checkpoint = Path(self.cfg.trainer.ckpt_path) / f"global_step_{self.global_step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        (checkpoint / "learner.json").write_text(
            json.dumps(
                {
                    "completed_update": self.policy_model.completed_update,
                    "installed_update": self.inference_engine_client.installed_update,
                }
            )
        )
        self._last_saved_step = self.global_step

    def load_checkpoints(self):
        checkpoint = Path(self.cfg.trainer.resume_path)
        saved = json.loads((checkpoint / "learner.json").read_text())
        self.policy_model.completed_update = saved["completed_update"]
        return saved["completed_update"], str(checkpoint)

    async def _run_training(self, training_input):
        self.policy_model.completed_update += 1
        responses = training_input["sequences"][:, -1].tolist()
        versions = [token - 10 for token in responses]
        self.policy_model.consumed.append((self.policy_model.completed_update, versions))
        self.all_timings["train_critic_and_policy"] = 0.001
        return {"learner/update": self.policy_model.completed_update}


def make_driver(
    *,
    interval=2,
    age=1,
    steps=5,
    eval_steps=100,
    report_step=True,
    stop_step=None,
    save_step=None,
    epochs=1,
    steps_per_epoch=None,
    runner_type=Runner,
    dynamic_sampling_type=None,
    driver_type=DriverWithCpuLearner,
):
    cfg = get_default_config()
    updates = {
        "trainer.fully_async.weight_sync_interval": interval,
        "trainer.fully_async.max_staleness_steps": age,
        "trainer.fully_async.num_parallel_generation_workers": 2,
        "trainer.fully_async.max_buffered_groups": 1,
        "trainer.fully_async.admission_stall_timeout": 3,
        "trainer.train_batch_size": 2,
        "trainer.policy_mini_batch_size": 2,
        "trainer.eval_batch_size": 1,
        "trainer.max_steps": steps,
        "trainer.epochs": epochs,
        "trainer.resume_mode": None,
        "trainer.placement.colocate_all": False,
        "trainer.training_metrics": True,
        "trainer.dump_eval_results": False,
        "trainer.algorithm.use_tis": False,
        "trainer.algorithm.dynamic_sampling.type": dynamic_sampling_type,
        "trainer.algorithm.policy_loss_type": "behavior_clip",
        "generator.enable_http_endpoint": True,
        "generator.async_engine": True,
        "generator.batched": False,
        "generator.n_samples_per_prompt": 2,
        "generator.eval_n_samples_per_prompt": 1,
    }
    if interval is None:
        del updates["trainer.fully_async.weight_sync_interval"]
    for path, value in updates.items():
        OmegaConf.update(cfg, path, value, force_add=True)
    engine = InferenceService()
    runner = runner_type(engine, report_step=report_step)
    trainer = driver_type(
        cfg=cfg,
        tracker=Tracker(),
        tokenizer=Tokenizer(),
        train_dataset=PromptRows(2 * (steps_per_epoch or steps)),
        eval_dataset=PromptRows(1),
        inference_engine_client=engine,
        trajectory_runner=runner,
        callbacks=[EvaluationCallback(eval_steps=eval_steps), LogAndStop(stop_step, save_step)],
    )
    trainer.policy_model = LearnerService()
    return trainer


@pytest.mark.asyncio
@pytest.mark.parametrize("report_step", [True, False])
@pytest.mark.parametrize(
    "interval,age,publications", [(None, 0, [0, 1, 2, 3, 4, 5]), (1, 1, [0, 1, 2, 3, 4, 5]), (2, 1, [0, 2, 4, 5])]
)
async def test_cadence_consumes_bounded_age_batches_and_evaluates_final_weights(
    interval, age, publications, report_step
):
    trainer = make_driver(interval=interval, age=age, report_step=report_step)
    await asyncio.wait_for(trainer._train_loop(), timeout=10)

    assert trainer.inference_engine_client.publications == publications
    assert trainer.trajectory_runner.evaluations == [(0, 0), (5, 5)]
    assert trainer.data_tracker.total_samples_consumed == 10
    assert [step for step, _ in trainer.policy_model.consumed] == [1, 2, 3, 4, 5]
    for step, versions in trainer.policy_model.consumed:
        assert all(0 <= step - 1 - version <= age for version in versions)
        logged = next(
            metrics
            for row_step, metrics in trainer.tracker.rows
            if row_step == step and "trainer/global_step" in metrics
        )
        assert logged["async/staleness_max"] == max(step - 1 - version for version in versions)
        if step not in publications:
            assert logged["timing/sync_weights"] == 0
            assert logged["async/performance/weight_sync_fraction"] == 0


@pytest.mark.asyncio
async def test_off_grid_evaluation_publishes_fresh_weights_without_shifting_grid():
    trainer = make_driver(eval_steps=3)
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    assert trainer.inference_engine_client.publications == [0, 2, 3, 4, 5]
    assert trainer.trajectory_runner.evaluations == [(0, 0), (3, 3), (5, 5)]
    step3 = next(metrics for step, metrics in trainer.tracker.rows if step == 3 and "trainer/global_step" in metrics)
    assert step3["timing/sync_weights"] == 0
    assert step3["timing/eval_weight_sync"] > 0


@pytest.mark.parametrize("interval,age", [(0, 1), (-1, 1), (1.5, 1), (True, 1), (2, 0), (3, 1), (1, -1)])
def test_impossible_or_invalid_cadence_is_rejected_before_training(interval, age):
    cfg = get_default_config()
    cfg.trainer.fully_async.weight_sync_interval = interval
    cfg.trainer.fully_async.max_staleness_steps = age
    with pytest.raises(ValueError, match="trainer.fully_async"):
        validate_cfg(cfg)


@pytest.mark.asyncio
async def test_off_grid_early_stop_preserves_stop_and_finalizes_completed_update():
    trainer = make_driver(stop_step=1)
    trainer.cfg.trainer.epochs = 2
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    assert trainer.inference_engine_client.publications == [0, 1]
    assert trainer.trajectory_runner.evaluations == [(0, 0), (1, 1)]
    assert [step for step, _ in trainer.policy_model.consumed] == [1]


@pytest.mark.asyncio
async def test_off_grid_checkpoint_resumes_on_global_grid_with_original_buffer_stamps(tmp_path):
    trainer = make_driver(stop_step=3, save_step=3)
    trainer.cfg.trainer.ckpt_path = str(tmp_path)
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    checkpoint = tmp_path / "global_step_3"
    saved = json.loads((checkpoint / "learner.json").read_text())
    assert saved == {"completed_update": 3, "installed_update": 2}

    resumed = make_driver()
    resumed.resume_mode = ResumeMode.LATEST
    resumed.cfg.trainer.resume_path = str(checkpoint)
    await asyncio.wait_for(resumed._train_loop(), timeout=10)
    assert resumed.inference_engine_client.publications == [3, 4, 5]
    assert resumed.trajectory_runner.evaluations == [(3, 3), (5, 5)]
    assert [step for step, _ in resumed.policy_model.consumed] == [4, 5]
    assert resumed.data_tracker.total_samples_consumed == 10
    for step, versions in resumed.policy_model.consumed:
        assert all(0 <= step - 1 - version <= 1 for version in versions)


@pytest.mark.asyncio
@pytest.mark.parametrize("max_steps", [3, 2])
async def test_resume_at_or_past_max_only_evaluates_checkpoint_weights(tmp_path, max_steps):
    checkpoint = tmp_path / "global_step_3"
    checkpoint.mkdir()
    (checkpoint / "learner.json").write_text(json.dumps({"completed_update": 3, "installed_update": 2}))
    resumed = make_driver(steps=max_steps)
    resumed.resume_mode = ResumeMode.LATEST
    resumed.cfg.trainer.resume_path = str(checkpoint)
    await asyncio.wait_for(resumed._train_loop(), timeout=10)
    assert resumed.inference_engine_client.publications == [3]
    assert resumed.trajectory_runner.evaluations == [(3, 3)]
    assert resumed.policy_model.consumed == []


@pytest.mark.asyncio
async def test_failed_publication_stops_before_evaluating_uninstalled_update():
    trainer = make_driver()
    trainer.policy_model.fail_update = 2
    try:
        with pytest.raises(RuntimeError, match="publication transport failed"):
            await asyncio.wait_for(trainer._train_loop(), timeout=10)
        assert trainer.inference_engine_client.publications == [0]
        assert trainer.trajectory_runner.evaluations == [(0, 0)]
        assert [step for step, _ in trainer.policy_model.consumed] == [1, 2]
    finally:
        for task in trainer._active_trajectory_tasks:
            task.cancel()
        await asyncio.gather(*trainer._active_trajectory_tasks, return_exceptions=True)


class DelayedFirstRunner(Runner):
    def __init__(self, engine, *, report_step):
        super().__init__(engine, report_step=report_step)
        self.release = asyncio.Event()
        self.delayed_uid = None

    async def run(self, request, **kwargs):
        batch = await super().run(request, **kwargs)
        if request["batch_metadata"].training_phase == "train" and self.delayed_uid is None:
            self.delayed_uid = request["trajectory_ids"][0].instance_id
            await self.release.wait()
        return batch


class ReleaseDelayedGroup(TrainerCallback):
    def __init__(self, runner):
        self.runner = runner

    def on_step_end(self, state, control, **kwargs):
        if state.global_step == 3:
            self.runner.release.set()
        return control


class UniformFirstRunner(Runner):
    async def run(self, request, **kwargs):
        batch = await super().run(request, **kwargs)
        if request["batch_metadata"].training_phase == "train" and len(self.generations) == 1:
            batch["rewards"] = [0.0] * len(batch["rewards"])
            batch["unshaped_rewards"] = list(batch["rewards"])
        return batch


@pytest.mark.asyncio
async def test_publication_ceiling_avoids_third_cohorts_without_changing_the_age_budget():
    trainer = make_driver()
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    versions = [version for _, version in trainer.trajectory_runner.generations]
    assert versions.count(0) == 4
    assert versions.count(2) == 4
    assert trainer.data_tracker.total_samples_consumed == 10
    assert trainer.inference_engine_client.publications == [0, 2, 4, 5]


@pytest.mark.asyncio
async def test_slow_crossing_group_is_retried_and_consumed_without_capacity_deadlock():
    trainer = make_driver(age=2, runner_type=DelayedFirstRunner)
    runner = trainer.trajectory_runner
    trainer.callback_handler.add_callback(ReleaseDelayedGroup(runner))
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    versions = [version for uid, version in runner.generations if uid == runner.delayed_uid]
    assert versions[0] == 0
    assert any(version >= 2 for version in versions[1:])
    assert trainer.data_tracker.total_samples_consumed == 10
    assert sum(metrics.get("async/rejected_count/stale", 0) for _, metrics in trainer.tracker.rows) >= 1
    for step, versions in trainer.policy_model.consumed:
        assert all(0 <= step - 1 - version <= 2 for version in versions)


@pytest.mark.asyncio
async def test_dynamic_filter_releases_publication_capacity_for_fresh_candidates():
    trainer = make_driver(runner_type=UniformFirstRunner, dynamic_sampling_type="filter")
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    assert trainer.data_tracker.total_samples_consumed == 10
    assert sum(metrics.get("async/dynamic_sampling/discarded_count", 0) for _, metrics in trainer.tracker.rows) >= 1
    assert trainer.trajectory_runner.evaluations == [(0, 0), (5, 5)]


@pytest.mark.asyncio
async def test_publication_capacity_survives_an_odd_epoch_boundary():
    trainer = make_driver(steps=6, epochs=2, steps_per_epoch=3)
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    assert trainer.inference_engine_client.publications == [0, 2, 4, 6]
    assert trainer.data_tracker.total_samples_consumed == 12
    assert [step for step, _ in trainer.policy_model.consumed] == [1, 2, 3, 4, 5, 6]
    assert trainer.trajectory_runner.evaluations == [(0, 0), (6, 6)]


class SyncStartupDriver(RayPPOTrainer):
    def init_weight_sync_state(self):
        pass


class StartupLearnerService(LearnerService):
    def async_run_ray_method(self, dispatch, method, *args):
        if method == "broadcast_to_inference_engines":
            (engine,) = args
            engine.installed_update = self.completed_update
            engine.publications.append(self.completed_update)
            return []
        return super().async_run_ray_method(dispatch, method)


class FrozenEvaluationRunner(Runner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observed_publications = []
        self.fail_pass = None

    async def run(self, request, **kwargs):
        if request["batch_metadata"].training_phase == "eval" and request["batch_metadata"].global_step == 0:
            self.observed_publications.append(list(self.engine.publications))
            if len(self.observed_publications) == self.fail_pass:
                raise RuntimeError("frozen evaluation failed")
        return await super().run(request, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize("driver_type", [SyncStartupDriver, DriverWithCpuLearner])
@pytest.mark.parametrize("step_wise", [False, True])
@pytest.mark.parametrize("repeats", [1, 3])
async def test_startup_repeats_preserve_frozen_weights_and_unique_dumps(
    tmp_path, monkeypatch, driver_type, step_wise, repeats
):
    trainer = make_driver(driver_type=driver_type, runner_type=FrozenEvaluationRunner)
    trainer.policy_model = StartupLearnerService()
    # Exercise the real startup and finalization sequence without subsequent optimizer work.
    trainer.cfg.trainer.epochs = 0
    trainer.cfg.trainer.initial_eval_repeat_count = repeats
    trainer.cfg.trainer.dump_eval_results = True
    trainer.cfg.trainer.step_wise_training = step_wise
    trainer.cfg.trainer.export_path = str(tmp_path)
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    assert trainer.inference_engine_client.publications == [0]
    assert trainer.trajectory_runner.evaluations == [(0, 0)] * repeats
    assert trainer.trajectory_runner.observed_publications == [[0]] * repeats
    assert trainer.policy_model.completed_update == 0
    assert trainer.policy_model.consumed == []
    rows = [
        metrics for step, metrics in trainer.tracker.rows if step == 0 and any(k.startswith("eval/") for k in metrics)
    ]
    assert len(rows) == 1
    root = tmp_path / "dumped_evals" / "global_step_0_evals"
    if repeats == 1:
        assert rows[0]["eval/all/avg_score"] == 0.0
        assert json.loads((root / "unknown.jsonl").read_text())["uid"] == "0"
        assert len(list(root.rglob("aggregated_results.jsonl"))) == 1
    else:
        assert all(key.startswith("eval/startup_pass_") for key in rows[0])
        for pass_index in range(repeats):
            namespace = f"startup_pass_{pass_index}"
            assert rows[0][f"eval/{namespace}/all/avg_score"] == 0.0
            row = json.loads((root / namespace / "unknown.jsonl").read_text())
            assert row["uid"] == "0"
            assert row["response_ids"] == [10]
        assert len(list(root.rglob("aggregated_results.jsonl"))) == repeats


@pytest.mark.asyncio
async def test_async_optimizer_starts_only_after_all_startup_passes(tmp_path):
    trainer = make_driver(steps=2, runner_type=FrozenEvaluationRunner)
    trainer.cfg.trainer.initial_eval_repeat_count = 3
    trainer.cfg.trainer.dump_eval_results = True
    trainer.cfg.trainer.export_path = str(tmp_path)
    await asyncio.wait_for(trainer._train_loop(), timeout=10)
    assert trainer.trajectory_runner.evaluations == [(0, 0)] * 3 + [(2, 2)]
    assert trainer.trajectory_runner.observed_publications == [[0]] * 3
    assert trainer.inference_engine_client.publications == [0, 2]
    assert [step for step, _ in trainer.policy_model.consumed] == [1, 2]


@pytest.mark.asyncio
@pytest.mark.parametrize("driver_type", [SyncStartupDriver, DriverWithCpuLearner])
async def test_startup_repeat_failure_prevents_training(tmp_path, monkeypatch, driver_type):
    trainer = make_driver(driver_type=driver_type, runner_type=FrozenEvaluationRunner)
    trainer.policy_model = StartupLearnerService()
    trainer.cfg.trainer.initial_eval_repeat_count = 3
    trainer.cfg.trainer.dump_eval_results = True
    trainer.cfg.trainer.export_path = str(tmp_path)
    trainer.trajectory_runner.fail_pass = 2
    monkeypatch.setattr("skyrl_train.trainer.ray.get", lambda refs: refs)
    with pytest.raises(RuntimeError, match="frozen evaluation failed"):
        await trainer._train_loop()
    assert trainer.inference_engine_client.publications == [0]
    assert trainer.policy_model.completed_update == 0
    assert trainer.trajectory_runner.generations == []
    assert len(list(tmp_path.rglob("aggregated_results.jsonl"))) == 1
    assert not any(key.startswith("eval/") for _, metrics in trainer.tracker.rows for key in metrics)


@pytest.mark.parametrize("override", ["0", "-1", "1.5", "true", "null"])
def test_initial_eval_rejects_invalid_repeat_recipe_override(override):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME, overrides=[f"trainer.initial_eval_repeat_count={override}"])
    with pytest.raises(ValueError, match="initial_eval_repeat_count"):
        validate_cfg(cfg)


@pytest.mark.parametrize("disabled", ["eval_before_train=false", "eval_interval=0", "dump_eval_results=false"])
def test_initial_eval_repeat_recipe_requires_evaluation_and_dumps(disabled):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name=DEFAULT_CONFIG_NAME, overrides=["trainer.initial_eval_repeat_count=3", f"trainer.{disabled}"]
        )
    with pytest.raises(ValueError, match="initial_eval_repeat_count"):
        validate_cfg(cfg)


@pytest.mark.parametrize("overrides,expected", [([], 1), (["trainer.initial_eval_repeat_count=3"], 3)])
def test_initial_eval_recipe_composes_and_validates(overrides, expected):
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=DEFAULT_CONFIG_NAME, overrides=overrides)
    validate_cfg(cfg)
    assert cfg.trainer.initial_eval_repeat_count == expected
