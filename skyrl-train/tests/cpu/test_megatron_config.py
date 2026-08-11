import pytest

from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def test_megatron_rejects_unsupported_router_replay():
    cfg = get_default_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.policy.fsdp_config.moe_router_replay = True

    with pytest.raises(ValueError, match="moe_router_replay.*fsdp.*fsdp2"):
        validate_cfg(cfg)
