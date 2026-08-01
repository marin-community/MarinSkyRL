"""Default values for EP and the FSDP fields it extends.

Every trainer role must expose the same disabled EP defaults and base FSDP2
defaults.

Run:
    uv run --isolated --extra dev pytest tests/cpu/test_ep_config_noop.py
"""

from skyrl_train.config.utils import get_default_config

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

BASE_FSDP_DEFAULTS = {
    "cpu_offload": False,
    "reshard_after_forward": True,
    "fsdp_size": -1,
}


def test_ep_and_base_fsdp_fields_parse_with_defaults():
    expected_defaults = BASE_FSDP_DEFAULTS | EP_FIELDS
    cfg = get_default_config()
    for model in ("policy", "ref", "critic"):
        fsdp = cfg.trainer[model].fsdp_config
        for k, v in expected_defaults.items():
            assert k in fsdp, f"trainer.{model}.fsdp_config missing {k}"
            assert fsdp[k] == v, f"trainer.{model}.fsdp_config.{k}={fsdp[k]!r}, expected {v!r}"
