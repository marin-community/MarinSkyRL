from unittest import mock

from omegaconf import OmegaConf
import pytest
from ci.pivot_grug_smoke import MODEL_REVISION, launch_config, validate_smoke_config
from cloud.iris.launch_config import load_launch_config
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def _megatron_replay_cfg():
    cfg = get_default_config()
    cfg.trainer.strategy = "megatron"
    cfg.trainer.logger = "console"
    cfg.trainer.policy.megatron_config.moe_router_replay = True
    return cfg


def test_megatron_router_replay_reaches_dcp_guard():
    """Enabled Megatron replay counts as active R3 capture at the DCP guard."""
    cfg = _megatron_replay_cfg()
    cfg.generator.inference_engine_decode_context_parallel_size = 2
    cfg.generator.inference_engine_tensor_parallel_size = 8

    with (
        mock.patch("transformers.AutoConfig.from_pretrained", side_effect=OSError("offline")),
        pytest.raises(AssertionError, match="decode context parallel.*R3 router capture"),
    ):
        validate_cfg(cfg)


def test_unsupported_strategy_rejected():
    cfg = get_default_config()
    cfg.trainer.strategy = "unknown"
    cfg.trainer.logger = "console"

    with pytest.raises(ValueError, match="Unsupported training strategy"):
        validate_cfg(cfg)


def test_megatron_router_replay_rejects_fused_router():
    cfg = _megatron_replay_cfg()
    # Struct mode blocks new dict keys; the launcher's Hydra `+` override allows
    # exactly this addition, so mirror it here.
    OmegaConf.set_struct(cfg, False)
    cfg.trainer.policy.megatron_config.transformer_config_kwargs.moe_router_fusion = True

    with pytest.raises(ValueError, match="moe_router_fusion"):
        validate_cfg(cfg)


@pytest.mark.parametrize("role", ["policy", "ref"])
def test_grug_smoke_rejects_context_parallel_sliding_window_before_launch(tmp_path, role):
    config = launch_config(
        "grug-cp-regression",
        str(tmp_path / "output"),
        str(tmp_path / "temporary"),
        "s3://marin-us-east-02a/preflight/grug",
        MODEL_REVISION,
    )
    # Packing passed the generic CP check but then failed inside TE attention on the GPUs.
    config.skyrl.trainer.use_sample_packing = True
    config.skyrl.trainer[role].megatron_config.context_parallel_size = 2
    path = tmp_path / "launch.yaml"
    OmegaConf.save(config, path)
    resolved = load_launch_config(path)

    with pytest.raises(ValueError, match=rf"trainer\.{role}\.megatron_config\.context_parallel_size=1"):
        validate_smoke_config(resolved.skyrl)
