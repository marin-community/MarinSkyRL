"""Eligibility and bundle shape of the dedicated STRICT_SPREAD policy placement group."""

import pytest
from omegaconf import OmegaConf

from skyrl_train.utils.utils import policy_spread_bundles, policy_strict_spread_eligible


def _make_cfg(
    *,
    colocate_all=False,
    policy_num_nodes=8,
    policy_num_gpus_per_node=4,
    policy_strict_spread_pg=True,
    use_kl_loss=False,
    use_kl_in_reward=False,
    policy_per_gpu_bundles=False,
):
    return OmegaConf.create(
        {
            "trainer": {
                "placement": {
                    "colocate_all": colocate_all,
                    "policy_num_nodes": policy_num_nodes,
                    "policy_num_gpus_per_node": policy_num_gpus_per_node,
                    "policy_strict_spread_pg": policy_strict_spread_pg,
                    "policy_per_gpu_bundles": policy_per_gpu_bundles,
                },
                "algorithm": {"use_kl_loss": use_kl_loss, "use_kl_in_reward": use_kl_in_reward},
            }
        }
    )


@pytest.mark.parametrize(
    ("overrides", "eligible"),
    [
        pytest.param({}, True, id="disaggregated-no-ref"),
        pytest.param({"policy_strict_spread_pg": False}, False, id="flag-off"),
        pytest.param({"use_kl_loss": True}, False, id="ref-model-via-kl-loss"),
        pytest.param({"use_kl_in_reward": True}, False, id="ref-model-via-kl-in-reward"),
        pytest.param({"colocate_all": True}, False, id="colocate-all"),
    ],
)
def test_policy_strict_spread_eligibility(overrides, eligible):
    assert policy_strict_spread_eligible(_make_cfg(**overrides)) is eligible


@pytest.mark.parametrize(
    ("per_gpu_bundles", "expected"),
    [
        # One whole-node bundle per policy node.
        pytest.param(False, [{"GPU": 4, "CPU": 4}] * 8, id="whole-node"),
        # One bundle per policy GPU, so len(bundles) == world_size engages the reordered-bundle path and
        # each actor resolves a distinct physical GPU (the GH200 device-collision fix).
        pytest.param(True, [{"GPU": 1, "CPU": 1}] * 32, id="per-gpu"),
    ],
)
def test_policy_spread_bundles_reserve_every_policy_gpu(per_gpu_bundles, expected):
    cfg = _make_cfg(policy_num_nodes=8, policy_num_gpus_per_node=4, policy_per_gpu_bundles=per_gpu_bundles)
    assert policy_spread_bundles(cfg) == expected
