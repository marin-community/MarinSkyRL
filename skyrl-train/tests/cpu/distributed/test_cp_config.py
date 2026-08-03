"""Stage-0 guarantee for the FSDP2 torch-native Context-Parallel (CP) port.

The context-parallel fields added to each `fsdp_config` block in
`ppo_base_config.yaml` (`context_parallel_size`, `cp_style`, `cp_rotate_method`)
must:

  (G0) parse with behavior-preserving defaults (CP disabled, `context_parallel_size == 1`)
       across all three roles (policy/ref/critic);
  (G1) be PURELY ADDITIVE — an all-defaults config with the three new keys removed is
       structurally identical to the pre-CP config (the provable guarantee that the
       default/production FSDP2 path is byte-identical post-change);
  (G2) when enabled (`context_parallel_size > 1`), enforce the mutual-exclusion /
       backend asserts in `_validate_cp_cfg` (FSDP2-only, ⊥ Ulysses, ring_sdpa only,
       valid rotate method, no sample packing, divides the role world size).

See notes/RL/skyrl/fsdp2_context_parallel_stages/{README,stage0_config_scaffold_scope}.md.

Run:
    uv run --isolated --group dev --extra cpu pytest tests/cpu/distributed/test_cp_config.py -v
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from tests.cpu.fsdp_config_assertions import TRAINER_MODEL_ROLES, assert_role_fsdp_defaults

# The baseline snapshots `get_default_config()` without the intentionally additive
# CP, grouped-mm, attention-backend, and DCP fields. Keeping the remaining defaults
# current makes this a CP no-op regression test rather than a stale whole-config lock.
# `resolve=False` preserves interpolations, so the comparison is HOME-/env-independent.
GOLDEN = Path(__file__).parent.parent / "data" / "ppo_base_pre_cp.yaml"

CP_FIELDS = {
    "context_parallel_size": 1,
    "cp_style": "ring_sdpa",
    "cp_rotate_method": "allgather",
}
# Stage-2 additive top-level trainer keys (not under fsdp_config). Like CP_FIELDS,
# these are purely additive and must be stripped before the structural-identity
# comparison against the pre-CP golden. Default "auto" preserves byte-identical
# behavior (G1).
STAGE2_TRAINER_FIELDS = {
    "attn_backend": "auto",
    # preflight_gate is a default-off reward gate unrelated to CP; stripped here
    # because it is additive and must not appear in the pre-CP golden diff.
    "preflight_gate": {
        "enabled": False,
        "min_reward": 0.25,
        "max_reward": 0.75,
        "num_trials": 256,
        "on_failure": "abort",
    },
}
# Additive generator key from the vLLM DCP port Stage 0 (flag-off no-op, default == 1).
# Like the CP fields, it is purely additive and must be stripped before the structural
# -identity comparison against the pre-CP golden (it is unrelated to CP — DCP is the
# rollout-side decode KV-cache shard).
STAGE0_DCP_GENERATOR_FIELDS = {
    "inference_engine_decode_context_parallel_size": 1,
}
# Additive MoE fsdp_config key (runtime grouped-mm MoE swap). Flag-off no-op
# (default == False) and unrelated to CP; it landed after the pre-CP golden was
# snapshotted, so — like the CP fields — it must be stripped from each role's
# fsdp_config before the structural-identity comparison against the golden.
MOE_FSDP_FIELDS = {
    "use_grouped_mm": False,
}


# ----------------------------------------------------------------------------- G0
def test_cp_fields_parse_with_defaults():
    """All three CP keys present, with disabled defaults, in every role's fsdp_config."""
    assert_role_fsdp_defaults(get_default_config(), CP_FIELDS)


def test_default_config_validates_noop():
    """validate_cfg must accept the all-defaults config (CP disabled => strict no-op).

    The minimal default config may still trip unrelated (batch-size/placement) asserts,
    so we only require that no *CP* assertion fires for the disabled default.
    """
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = get_default_config()
    # _validate_cp_cfg in isolation must be a clean no-op at defaults.
    _validate_cp_cfg(cfg)


# ----------------------------------------------------------------------------- G1
def test_all_defaults_is_structurally_identical_to_baseline():
    """Removing the additive fields must reproduce the current no-CP baseline.

    Proves the default (production) path is byte-identical post-change.
    """
    container = OmegaConf.to_container(get_default_config(), resolve=False, throw_on_missing=False)
    for role in TRAINER_MODEL_ROLES:
        fsdp = container["trainer"][role]["fsdp_config"]
        for k in (*CP_FIELDS, *MOE_FSDP_FIELDS):  # strip the additive keys -> should reproduce pre-CP shape
            fsdp.pop(k, None)
    for k in STAGE2_TRAINER_FIELDS:  # strip Stage-2 additive top-level trainer keys
        container["trainer"].pop(k, None)
    for k in STAGE0_DCP_GENERATOR_FIELDS:  # strip DCP Stage-0 additive generator key
        container["generator"].pop(k, None)
    golden = OmegaConf.to_container(OmegaConf.load(GOLDEN), resolve=False, throw_on_missing=False)
    assert container == golden, "default config drifted from the no-CP baseline"


def test_diff_is_exactly_the_additive_fsdp_keys_x_three_roles():
    """The fsdp_config delta vs the golden is EXACTLY the additive keys (3 CP + grouped-mm) per role."""
    current = OmegaConf.to_container(get_default_config(), resolve=False, throw_on_missing=False)
    golden = OmegaConf.to_container(OmegaConf.load(GOLDEN), resolve=False, throw_on_missing=False)
    expected_added = set(CP_FIELDS) | set(MOE_FSDP_FIELDS)
    for role in TRAINER_MODEL_ROLES:
        cur_fsdp = current["trainer"][role]["fsdp_config"]
        gold_fsdp = golden["trainer"][role]["fsdp_config"]
        added = set(cur_fsdp) - set(gold_fsdp)
        assert added == expected_added, (
            f"trainer.{role}.fsdp_config added keys {sorted(added)}, expected {sorted(expected_added)}"
        )
        # And the added keys carry the disabled defaults.
        for k, v in {**CP_FIELDS, **MOE_FSDP_FIELDS}.items():
            assert cur_fsdp[k] == v
    # The only new top-level trainer keys are the Stage-2 additive ones (attn_backend).
    added_trainer = set(current["trainer"]) - set(golden["trainer"])
    assert added_trainer == set(STAGE2_TRAINER_FIELDS), (
        f"trainer added top-level keys {sorted(added_trainer)}, expected {sorted(STAGE2_TRAINER_FIELDS)}"
    )
    for k, v in STAGE2_TRAINER_FIELDS.items():
        assert current["trainer"][k] == v


# ----------------------------------------------------------------------------- G2
def _cp_enabled_config(role: str = "policy", cp_size: int = 2):
    """Full default config with CP enabled on `role` and a world size that admits it.

    Sets fsdp2 + a CP-compatible world size + sample packing off so the only failing
    assertion under test is the one each parametrized case deliberately introduces.
    """
    cfg = get_default_config()
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.use_sample_packing = False
    # Give every role a world size divisible by cp_size (default 4 gpus/node already is).
    for r in TRAINER_MODEL_ROLES:
        cfg.trainer[r].sequence_parallel_size = 1
    cfg.trainer[role].fsdp_config.context_parallel_size = cp_size
    cfg.trainer[role].fsdp_config.cp_style = "ring_sdpa"
    cfg.trainer[role].fsdp_config.cp_rotate_method = "allgather"
    return cfg


def test_cp_enabled_valid_config_passes():
    """A correctly-configured CP-enabled config passes _validate_cp_cfg."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    _validate_cp_cfg(_cp_enabled_config())


def test_cp_rejects_ulysses_combo():
    """CP enabled + sequence_parallel_size > 1 (Ulysses) => mutual-exclusion assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = _cp_enabled_config()
    cfg.trainer.policy.sequence_parallel_size = 2
    with pytest.raises(AssertionError, match="mutually"):
        _validate_cp_cfg(cfg)


def test_cp_rejects_megatron_strategy():
    """CP enabled + strategy=megatron => FSDP2-only assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = _cp_enabled_config()
    cfg.trainer.strategy = "megatron"
    with pytest.raises(AssertionError, match="fsdp2"):
        _validate_cp_cfg(cfg)


def test_cp_rejects_ring_flash_attn_style():
    """CP enabled + cp_style=ring_flash_attn => unsupported-style assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = _cp_enabled_config()
    cfg.trainer.policy.fsdp_config.cp_style = "ring_flash_attn"
    with pytest.raises(AssertionError, match="cp_style"):
        _validate_cp_cfg(cfg)


def test_cp_rejects_bad_rotate_method():
    """CP enabled + invalid cp_rotate_method => invalid-rotate assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = _cp_enabled_config()
    cfg.trainer.policy.fsdp_config.cp_rotate_method = "p2p"
    with pytest.raises(AssertionError, match="cp_rotate_method"):
        _validate_cp_cfg(cfg)


def test_cp_rejects_sample_packing():
    """CP enabled + sample packing on => packed-varlen-deferred assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = _cp_enabled_config()
    cfg.trainer.use_sample_packing = True
    with pytest.raises(AssertionError, match="sample packing"):
        _validate_cp_cfg(cfg)


def test_cp_rejects_indivisible_world_size():
    """CP size that does not divide the role world size => divisibility assert."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = _cp_enabled_config(cp_size=3)  # default 4 gpus/node, 1 node -> 4 % 3 != 0
    cfg.trainer.placement.policy_num_gpus_per_node = 4
    cfg.trainer.placement.policy_num_nodes = 1
    with pytest.raises(AssertionError, match="divide"):
        _validate_cp_cfg(cfg)


@pytest.mark.parametrize("role", TRAINER_MODEL_ROLES)
def test_cp_mutual_exclusion_enforced_per_role(role):
    """The Ulysses mutual-exclusion assert fires for each role independently."""
    pytest.importorskip("hydra")
    from skyrl_train.utils.utils import _validate_cp_cfg

    cfg = _cp_enabled_config(role=role)
    cfg.trainer[role].sequence_parallel_size = 2
    with pytest.raises(AssertionError, match="mutually"):
        _validate_cp_cfg(cfg)
