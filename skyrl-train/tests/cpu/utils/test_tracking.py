from unittest.mock import Mock

from skyrl_train.utils.tracking import Tracking


def test_wandb_shared_mode_logs_with_custom_global_step() -> None:
    run = Mock()
    tracker = Tracking.__new__(Tracking)
    tracker.logger = {"wandb": run}

    tracker.log({"reward": 0.5}, step=7, commit=True)

    run.log.assert_called_once_with(data={"reward": 0.5, "trainer/global_step": 7}, commit=True)


def test_finish_is_idempotent() -> None:
    run = Mock()
    tracker = Tracking.__new__(Tracking)
    tracker.logger = {"wandb": run}

    tracker.finish()
    tracker.finish()

    run.finish.assert_called_once_with(exit_code=0)
