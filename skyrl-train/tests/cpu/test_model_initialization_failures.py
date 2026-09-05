import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger
from ray.exceptions import GetTimeoutError

import skyrl_train.trainer as trainer_module
from skyrl_train.config.utils import get_default_config
from skyrl_train.entrypoints.fully_async import AsyncPPOExp
from skyrl_train.entrypoints.main_base import BasePPOExp
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.utils.tracking import Tracking


@pytest.mark.parametrize("experiment_type", [BasePPOExp, AsyncPPOExp])
@pytest.mark.parametrize("failure", [None, "setup", "train"])
def test_experiment_finishes_tracking_before_return(experiment_type, failure, monkeypatch):
    monkeypatch.delenv("SKYRL_TELEMETRY_ENDPOINT", raising=False)
    finished = []

    class WandbService:
        def finish(self, exit_code):
            finished.append(exit_code)

    tracker = object.__new__(Tracking)
    tracker.logger = {"wandb": WandbService()}

    async def train():
        if failure == "train":
            raise RuntimeError("training failed")

    async def shutdown():
        pass

    class Experiment(experiment_type):
        def _setup_trainer(self):
            # CPU workload at the model/Ray boundary; retain the real experiment lifecycle.
            self.tracker = tracker
            if failure == "setup":
                raise RuntimeError("setup failed")
            return SimpleNamespace(train=train, shutdown=shutdown)

    experiment = object.__new__(Experiment)
    experiment.cfg = get_default_config()
    if failure is None:
        experiment.run()
    else:
        with pytest.raises(RuntimeError, match="failed"):
            experiment.run()
    assert finished == [0 if failure is None else 1]
    tracker.finish()
    assert finished == [0 if failure is None else 1]


class _UnpickleableError(RuntimeError):
    def __reduce__(self):
        raise pickle.PicklingError("exception cannot be pickled")


def test_model_initialization_timeout_logs_and_kills_actors(monkeypatch):
    trainer = object.__new__(RayPPOTrainer)
    trainer._kill_ray_actors = Mock()
    get = Mock(side_effect=GetTimeoutError("workers still downloading"))
    monkeypatch.setattr(trainer_module.ray, "get", get)
    monkeypatch.setattr(trainer_module, "time", SimpleNamespace(monotonic=lambda: 100.0), raising=False)
    messages = []
    sink_id = logger.add(messages.append, level="ERROR")

    try:
        with pytest.raises(RuntimeError, match="timed out after 3600 seconds"):
            trainer._wait_for_setup_phase(
                ["policy-worker-ref"],
                deadline=3700.0,
                phase="policy/ref/critic model initialization",
            )
    finally:
        logger.remove(sink_id)

    get.assert_called_once_with(["policy-worker-ref"], timeout=3600.0)
    trainer._kill_ray_actors.assert_called_once_with()
    assert len(messages) == 1
    assert messages[0].record["level"].name == "ERROR"


@pytest.mark.asyncio
async def test_startup_failure_still_runs_trainer_shutdown():
    events = []
    trainer = object.__new__(RayPPOTrainer)
    trainer._shutdown_complete = False

    async def fail_startup():
        events.append("startup")
        raise RuntimeError("runner failed to start")

    async def teardown():
        events.append("teardown")

    trainer._startup_trajectory_runner = fail_startup
    trainer._teardown = teardown

    with pytest.raises(RuntimeError, match="runner failed to start"):
        await trainer.train()

    assert events == ["startup", "teardown"]


@pytest.mark.asyncio
async def test_trainer_shutdown_is_idempotent():
    events = []
    trainer = object.__new__(RayPPOTrainer)
    trainer._shutdown_complete = False

    async def teardown():
        events.append("teardown")

    trainer._teardown = teardown

    await trainer.shutdown()
    await trainer.shutdown()

    assert events == ["teardown"]


@pytest.mark.asyncio
async def test_training_failure_log_record_does_not_contain_exception_object():
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 12
    trainer.trajectory_runner = SimpleNamespace(startup=AsyncMock())
    trainer._train_loop = AsyncMock(side_effect=_UnpickleableError("GPU worker ran out of memory"))
    trainer._cancel_trajectory_tasks = AsyncMock()
    trainer._teardown = AsyncMock()
    messages = []
    sink_id = logger.add(messages.append, level="ERROR")

    try:
        with pytest.raises(_UnpickleableError, match="GPU worker ran out of memory"):
            await trainer.train()
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    record = messages[0].record
    assert record["level"].name == "ERROR"
    assert record["exception"] is None
    assert "_UnpickleableError: GPU worker ran out of memory" in record["message"]
    pickle.dumps(record)
