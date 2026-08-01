"""Default values for EP configuration.

Every trainer role must expose the same disabled EP defaults.

Run:
    uv run --isolated --extra dev pytest tests/cpu/test_ep_config_defaults.py
"""

from skyrl_train.config.utils import get_default_config
from tests.cpu.fsdp_config_assertions import assert_role_fsdp_defaults

EP_FIELDS = {
    "expert_model_parallel_size": 1,
    "expert_tensor_parallel_size": 1,
    "moe_token_dispatcher_type": "alltoall",
    "moe_router_replay": False,
    "moe_grouped_gemm": False,
    "ep_comm_backend": "torch",
    "deepep_num_sms": 20,
    "deepep_token_chunk_size": None,
}


def test_ep_fields_parse_with_defaults():
    assert_role_fsdp_defaults(get_default_config(), EP_FIELDS)
