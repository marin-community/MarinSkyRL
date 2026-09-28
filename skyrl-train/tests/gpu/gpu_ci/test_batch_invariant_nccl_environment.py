import os
from unittest.mock import patch

import pytest

from skyrl_train.env_vars import VLLM_BATCH_INVARIANT_ENV
from skyrl_train.utils.utils import prepare_runtime_environment
from tests.cpu.util import example_dummy_config


@pytest.mark.vllm
def test_batch_invariant_nccl_settings_match_installed_vllm(monkeypatch):
    monkeypatch.setattr("skyrl_train.utils.utils.peer_access_supported", lambda **_: True)
    cfg = example_dummy_config()
    clean_env = {
        name: value
        for name, value in os.environ.items()
        if not name.startswith("NCCL_") and name != VLLM_BATCH_INVARIANT_ENV
    }

    with patch.dict(os.environ, clean_env, clear=True):
        default_env = prepare_runtime_environment(cfg)
        cfg.trainer.algorithm.batch_invariant = True
        invariant_env = prepare_runtime_environment(cfg)
        skyrl_nccl = {
            name: value
            for name, value in invariant_env.items()
            if name.startswith("NCCL_") and default_env.get(name) != value
        }

        os.environ[VLLM_BATCH_INVARIANT_ENV] = "1"
        from vllm.model_executor.determinism.batch_invariant import override_envs_for_invariance

        override_envs_for_invariance()
        vllm_nccl = {name: value for name, value in os.environ.items() if name.startswith("NCCL_")}

    assert skyrl_nccl == vllm_nccl
