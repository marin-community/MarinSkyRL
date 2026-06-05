"""Stage-0 guarantee for the EP / router-replay port.

The expert-parallel / router-replay fields added to each `fsdp_config` block in
`ppo_base_config.yaml` must (a) parse with behavior-preserving defaults and
(b) be PURELY ADDITIVE — i.e. an all-defaults config with the new keys removed is
structurally identical to the pre-EP config. (b) is the provable guarantee that the
default (production) FSDP2 path is unchanged by Stage 0.

See notes/skyrl/fsdp2_ep_router_replay_port_plan.md (Stage 0).

Run:
    uv run --isolated --extra dev pytest tests/cpu/test_ep_config_noop.py
"""

from pathlib import Path

from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config

# The pre-EP golden was snapshotted from `get_default_config()` on the commit
# immediately before the EP keys were added (resolve=False, so interpolations are
# preserved verbatim and the comparison is HOME-/env-independent).
GOLDEN = Path(__file__).parent / "data" / "ppo_base_pre_ep.yaml"

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
    cfg = get_default_config()
    for model in ("policy", "ref", "critic"):
        fsdp = cfg.trainer[model].fsdp_config
        for k, v in EP_FIELDS.items():
            assert k in fsdp, f"trainer.{model}.fsdp_config missing {k}"
            assert fsdp[k] == v, f"trainer.{model}.fsdp_config.{k}={fsdp[k]!r}, expected {v!r}"


def test_all_defaults_is_structurally_identical_to_pre_ep():
    """Removing the new EP keys must reproduce the exact pre-EP config tree.

    Proves the default (production) path is byte-identical post-change.
    """
    container = OmegaConf.to_container(get_default_config(), resolve=False, throw_on_missing=False)
    for model in ("policy", "ref", "critic"):
        fsdp = container["trainer"][model]["fsdp_config"]
        for k in EP_FIELDS:  # strip the additive keys -> should reproduce pre-EP shape
            fsdp.pop(k, None)
    golden = OmegaConf.to_container(OmegaConf.load(GOLDEN), resolve=False, throw_on_missing=False)
    assert container == golden, "default config drifted from the pre-EP golden baseline"
