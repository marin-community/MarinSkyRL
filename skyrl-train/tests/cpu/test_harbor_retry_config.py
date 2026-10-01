"""Harbor retry policy preserves terminal results."""

from omegaconf import OmegaConf
from taskcompendium.rollout import RolloutFailure

from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings


def test_passthrough_exceptions_are_never_retried():
    cfg = OmegaConf.create(
        {
            "harbor": {
                "passthrough_exceptions": ["AgentTimeoutError"],
                "exclude_exceptions": ["VerifierTimeoutError"],
            }
        }
    )

    settings = HarborTaskSettings.from_config(cfg)

    for name in (
        "AgentTimeoutError",
        "OutputLengthExceededError",
        "TurnCapExhaustedError",
        "VerifierTimeoutError",
    ):
        assert settings.retry_delay(RolloutFailure(name), retries=0) is None
    assert settings.retry_delay(RolloutFailure("EnvironmentStartTimeoutError"), retries=0) == 1.0
