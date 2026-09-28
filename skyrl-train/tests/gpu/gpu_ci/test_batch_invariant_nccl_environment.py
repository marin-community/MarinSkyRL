import os
from unittest.mock import patch

import pytest

from skyrl_train.batch_invariant import BATCH_INVARIANT_NCCL_ENV


@pytest.mark.vllm
def test_batch_invariant_nccl_settings_match_installed_vllm():
    from vllm.model_executor.determinism.batch_invariant import override_envs_for_invariance

    with patch.dict(os.environ, {}, clear=True):
        override_envs_for_invariance()
        vllm_nccl = {name: value for name, value in os.environ.items() if name.startswith("NCCL_")}

    assert BATCH_INVARIANT_NCCL_ENV == vllm_nccl
