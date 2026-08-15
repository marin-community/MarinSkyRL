import pickle
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger
from ray.exceptions import GetTimeoutError

import skyrl_train.trainer as trainer_module
from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.trainer import RayPPOTrainer


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
async def test_training_failure_log_record_does_not_contain_exception_object():
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer.global_step = 12
    trainer.trajectory_runner = SimpleNamespace(startup=AsyncMock())
    trainer._maybe_enable_rollout_fanout = Mock()
    trainer._train_loop = AsyncMock(side_effect=_UnpickleableError("GPU worker ran out of memory"))
    trainer._cancel_trajectory_tasks = Mock()
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
