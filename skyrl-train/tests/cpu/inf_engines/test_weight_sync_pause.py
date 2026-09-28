import pytest

from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


@pytest.mark.parametrize("backend,local", [("sglang", True), ("vllm", False)])
def test_non_default_pause_policy_rejects_backend_without_pause(backend, local):
    cfg = get_default_config()
    cfg.generator.backend = backend
    cfg.generator.run_engines_locally = local
    cfg.generator.weight_sync_pause.mode = "keep"

    with pytest.raises(ValueError, match="requires local vLLM"):
        validate_cfg(cfg)
