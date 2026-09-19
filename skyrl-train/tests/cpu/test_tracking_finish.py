from unittest.mock import AsyncMock, Mock

import pytest

from skyrl_train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl_train.utils.tracking import Tracking


def test_tracking_finish_flushes_once_with_exit_status():
    tracker = object.__new__(Tracking)
    run = Mock()
    tracker.logger = {"wandb": run}
    tracker._finished = False

    tracker.finish(exit_code=1)
    tracker.finish(exit_code=0)

    run.finish.assert_called_once_with(exit_code=1)


@pytest.mark.parametrize("fails", [False, True])
@pytest.mark.asyncio
async def test_fully_async_train_finishes_tracking_after_shutdown(fails):
    trainer = object.__new__(FullyAsyncRayPPOTrainer)
    trainer._async_observations_enabled = False
    trainer._startup_trajectory_runner = AsyncMock()
    trainer._train_loop = AsyncMock(side_effect=RuntimeError("failed") if fails else None)
    trainer._cancel_trajectory_tasks = Mock()
    trainer.shutdown = AsyncMock()
    trainer.tracker = Mock()

    if fails:
        with pytest.raises(RuntimeError, match="failed"):
            await trainer.train()
    else:
        await trainer.train()

    trainer.shutdown.assert_awaited_once()
    trainer.tracker.finish.assert_called_once_with(exit_code=1 if fails else 0)
