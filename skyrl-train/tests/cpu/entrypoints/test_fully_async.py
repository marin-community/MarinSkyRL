import pytest
from omegaconf import OmegaConf

from skyrl_train.config.trajectory_runner_capabilities import TrajectoryRunnerMode
from skyrl_train.entrypoints.fully_async import trajectory_runner_mode


def _config(*, http: bool, multi_turn: bool):
    return OmegaConf.create(
        {
            "generator": {
                "enable_http_endpoint": http,
                "use_conversation_multi_turn": multi_turn,
            }
        }
    )


def test_direct_async_uses_exact_engine_runner():
    assert trajectory_runner_mode(_config(http=False, multi_turn=False)) is TrajectoryRunnerMode.SKYRL_GYM


def test_http_async_keeps_text_runner():
    assert trajectory_runner_mode(_config(http=True, multi_turn=True)) is TrajectoryRunnerMode.FULLY_ASYNC_SKYRL_GYM


@pytest.mark.parametrize(("http", "multi_turn"), [(True, False), (False, True)])
def test_async_runner_rejects_mixed_transport_settings(http, multi_turn):
    with pytest.raises(ValueError, match="require both false"):
        trajectory_runner_mode(_config(http=http, multi_turn=multi_turn))
