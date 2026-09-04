from unittest import mock

import pytest
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def test_megatron_dcp_reports_unsupported_router_replay():
    cfg = get_default_config()
    # The default logger is wandb, and validate_cfg asserts WANDB_API_KEY long before it reaches the
    # backend check -- so without this the test would pass on an unrelated failure.
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.policy.fsdp_config.moe_router_replay = True
    cfg.generator.inference_engine_decode_context_parallel_size = 2
    cfg.generator.inference_engine_tensor_parallel_size = 8

    with (
        mock.patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")),
        pytest.raises(ValueError, match="moe_router_replay.*fsdp.*fsdp2"),
    ):
        validate_cfg(cfg)


def test_validate_cfg_rejects_policy_train_spans_with_the_megatron_strategy():
    """🚨 Through the PUBLIC validator, on a real configuration.

    The helper is exercised in test_policy_train_spans, but nothing drove `validate_cfg` itself, so
    a dead call in front of it -- `if False: _validate_spans_backend(cfg)` -- accepted the
    combination with the suite green. That restores the failure the check was added to prevent: a
    Megatron run takes the flag, configures worker telemetry, trains normally, and publishes no tree
    at all while every health signal reads clean.

    ⚠️ A two-field mock config would fail earlier on unrelated required fields and a bare
    `raises(ValueError)` would then pass for the wrong reason, so this uses the real default config
    and matches the specific message.
    """
    import pytest
    from skyrl_train.config.utils import get_default_config
    from skyrl_train.utils.utils import validate_cfg

    cfg = get_default_config()
    # The default logger is wandb, and validate_cfg asserts WANDB_API_KEY long before it reaches the
    # backend check -- so without this the test would pass on an unrelated failure.
    cfg.trainer.logger = "console"
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy_train_spans = True
    with pytest.raises(ValueError, match="policy_train_spans is not supported"):
        validate_cfg(cfg)

    # The two negatives, so the guard cannot pass by rejecting everything.
    cfg.trainer.policy_train_spans = False
    validate_cfg(cfg)

    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.policy_train_spans = True
    validate_cfg(cfg)
