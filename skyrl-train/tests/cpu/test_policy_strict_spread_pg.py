"""
Unit tests for the dedicated STRICT_SPREAD policy placement-group computation.

These are pure (Ray-free) checks of the eligibility predicate and the
whole-node bundle list that the policy placement group is built from. They
assert that:
  - the feature is OFF by default (every existing run unchanged),
  - it only engages for a disaggregated, no-ref run with the flag set,
  - it stays OFF for the use_ref_model=True case and the colocate_all case,
  - the bundle list reserves exactly `policy_num_nodes` whole-node bundles.

uv run --isolated --extra dev pytest tests/cpu/test_policy_strict_spread_pg.py
"""

from omegaconf import OmegaConf

from skyrl_train.utils.utils import policy_strict_spread_eligible, policy_spread_bundles


def _make_cfg(
    *,
    colocate_all=False,
    policy_num_nodes=8,
    policy_num_gpus_per_node=4,
    policy_strict_spread_pg=True,
    use_kl_loss=False,
    use_kl_in_reward=False,
    include_flag=True,
):
    """Build a minimal config exercising only the placement/algorithm fields the
    eligibility predicate and bundle computation read. Avoids the hydra-backed
    full config loader so the test runs without the full SkyRL training deps."""
    placement = {
        "colocate_all": colocate_all,
        "policy_num_nodes": policy_num_nodes,
        "policy_num_gpus_per_node": policy_num_gpus_per_node,
    }
    if include_flag:
        placement["policy_strict_spread_pg"] = policy_strict_spread_pg
    return OmegaConf.create(
        {
            "trainer": {
                "placement": placement,
                "algorithm": {
                    "use_kl_loss": use_kl_loss,
                    "use_kl_in_reward": use_kl_in_reward,
                },
            }
        }
    )


def _disaggregated_no_ref_cfg(policy_num_nodes=8, policy_num_gpus_per_node=4, flag=True):
    """80B-style config: disaggregated, no ref model, flag enabled."""
    return _make_cfg(
        policy_num_nodes=policy_num_nodes,
        policy_num_gpus_per_node=policy_num_gpus_per_node,
        policy_strict_spread_pg=flag,
    )


def test_default_config_flag_is_off():
    """When the flag is absent from the config -> not eligible (default off)."""
    cfg = _make_cfg(include_flag=False)
    assert policy_strict_spread_eligible(cfg) is False


def test_eligible_disaggregated_no_ref_with_flag():
    """80B production case: disaggregated + no ref + flag -> eligible."""
    cfg = _disaggregated_no_ref_cfg()
    assert policy_strict_spread_eligible(cfg) is True


def test_not_eligible_without_flag():
    """Same disaggregated no-ref run but flag off -> NOT eligible (legacy path)."""
    cfg = _disaggregated_no_ref_cfg(flag=False)
    assert policy_strict_spread_eligible(cfg) is False


def test_not_eligible_with_ref_via_kl_loss():
    """use_ref_model True (via use_kl_loss) -> NOT eligible even with flag."""
    cfg = _disaggregated_no_ref_cfg()
    cfg.trainer.algorithm.use_kl_loss = True
    assert policy_strict_spread_eligible(cfg) is False


def test_not_eligible_with_ref_via_kl_in_reward():
    """use_ref_model True (via use_kl_in_reward) -> NOT eligible even with flag."""
    cfg = _disaggregated_no_ref_cfg()
    cfg.trainer.algorithm.use_kl_in_reward = True
    assert policy_strict_spread_eligible(cfg) is False


def test_not_eligible_when_colocate_all():
    """colocate_all True -> NOT eligible (single shared PG handles placement)."""
    cfg = _disaggregated_no_ref_cfg()
    cfg.trainer.placement.colocate_all = True
    assert policy_strict_spread_eligible(cfg) is False


def test_bundle_count_matches_policy_nodes_80b():
    """80B: 8 nodes x 4 GPU -> 8 whole-node bundles, 32 GPU total."""
    cfg = _disaggregated_no_ref_cfg(policy_num_nodes=8, policy_num_gpus_per_node=4)
    bundles = policy_spread_bundles(cfg)
    assert len(bundles) == 8
    assert all(b == {"GPU": 4, "CPU": 4} for b in bundles)
    total_gpus = sum(b["GPU"] for b in bundles)
    assert total_gpus == 32
    # Disjoint-fit check on the 24-node / 96-GPU cluster:
    # policy 32 GPU (8 nodes) + inference 16 engines x TP4 = 64 GPU (16 nodes) = 96 GPU / 24 nodes.
    num_inference_gpus = 16 * 4
    assert total_gpus + num_inference_gpus == 24 * 4

def test_bundle_count_matches_policy_nodes_a3_8b():
    """a3 8B shape: 2 nodes x 4 GPU -> 2 whole-node bundles, 8 GPU total.

    (a3 runs leave the flag OFF, so this PG is never actually built for them;
    this only asserts the bundle math is correct if a no-ref run ever opts in.)
    """
    cfg = _disaggregated_no_ref_cfg(policy_num_nodes=2, policy_num_gpus_per_node=4)
    bundles = policy_spread_bundles(cfg)
    assert len(bundles) == 2
    assert sum(b["GPU"] for b in bundles) == 8
