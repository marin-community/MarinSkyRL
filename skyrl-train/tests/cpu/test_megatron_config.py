from unittest import mock

import pytest
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def test_megatron_dcp_reports_unsupported_router_replay():
    cfg = get_default_config()
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
