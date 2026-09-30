import json
import os
import pickle
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from loguru import logger
from ray.exceptions import GetTimeoutError

from marinskyrl.environment_contract import DEBUG_ARTIFACT_DIR_ENV
import skyrl_train.trainer as trainer_module
from skyrl_train.trainer import RayPPOTrainer


class _UnpickleableError(RuntimeError):
    def __reduce__(self):
        raise pickle.PicklingError("exception cannot be pickled")


def test_model_initialization_timeout_raises_and_kills_actors(monkeypatch):
    trainer = object.__new__(RayPPOTrainer)
    killed = []
    waits = []
    trainer._kill_ray_actors = lambda: killed.append(True)

    def get(refs, timeout):
        waits.append(timeout)
        raise GetTimeoutError("workers still downloading")

    monkeypatch.setattr(trainer_module.ray, "get", get)
    monkeypatch.setattr(trainer_module, "time", SimpleNamespace(monotonic=lambda: 100.0), raising=False)

    with pytest.raises(RuntimeError, match="timed out after 3600 seconds"):
        trainer._wait_for_setup_phase(
            ["policy-worker-ref"],
            deadline=3700.0,
            phase="policy/ref/critic model initialization",
        )

    # The phase waits only for the budget left before the shared deadline.
    assert waits == [3600.0]
    assert killed == [True]


@pytest.mark.asyncio
async def test_startup_failure_still_runs_trainer_shutdown():
    events = []
    trainer = object.__new__(RayPPOTrainer)
    trainer._shutdown_complete = False
    trainer.global_step = 0
    trainer._distillation_runtime = None
    trainer.context = SimpleNamespace(close=AsyncMock())

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
    trainer.context = SimpleNamespace(close=AsyncMock())

    async def teardown():
        events.append("teardown")

    trainer._teardown = teardown

    await trainer.shutdown()
    await trainer.shutdown()

    assert events == ["teardown"]


@pytest.mark.asyncio
async def test_training_failure_log_record_does_not_contain_exception_object():
    trainer = object.__new__(RayPPOTrainer)
    trainer.global_step = 12
    trainer._distillation_runtime = None
    trainer.context = SimpleNamespace(close=AsyncMock())
    trainer.trajectory_runner = SimpleNamespace(startup=AsyncMock())
    trainer._train_loop = AsyncMock(side_effect=_UnpickleableError("GPU worker ran out of memory"))
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


def test_training_failure_receipt_survives_blocked_teacher_shutdown(tmp_path):
    # Use a separate process because the teardown guard must terminate even when
    # a teacher close blocks the event loop and cannot observe cancellation.
    script = textwrap.dedent("""
        import asyncio
        import threading

        from skyrl_train.trainer import RayPPOTrainer

        class StuckTeacherRuntime:
            async def start(self):
                pass

            async def close(self):
                threading.Event().wait()

        class FailedTrainer(RayPPOTrainer):
            def __init__(self):
                self.global_step = 0
                self._rollout_spans_enabled = False
                self._expert_block_sync = None
                self._distillation_runtime = StuckTeacherRuntime()

            async def _startup_trajectory_runner(self):
                pass

            async def _train_loop(self):
                raise ValueError("teacher scoring rejected the batch")

            @staticmethod
            def _start_exit_watchdog(timeout=120):
                RayPPOTrainer._start_exit_watchdog(timeout=0.1)

        asyncio.run(FailedTrainer().train())
    """)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, DEBUG_ARTIFACT_DIR_ENV: str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 1, result.stderr
    receipts = list((tmp_path / "outcomes").glob("skyrl-trainer.*.exception.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["exception_type"] == "builtins.ValueError"
    assert receipt["message"] == "teacher scoring rejected the batch"
    assert "_train_loop" in receipt["traceback"]
