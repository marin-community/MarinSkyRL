from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from skyrl_gym.envs.aime.env import AIMEEnv
from skyrl_train.entrypoints.non_agentic_probe_inputs import reward_extras


def test_actual_aime_constructor_receives_original_gold_channels():
    row = {"reward_spec": {"ground_truth": "42", "type": "math"}, "reward_model": {"ground_truth": "42"}}
    original = deepcopy(row)
    extras = reward_extras(row)
    environment = AIMEEnv(OmegaConf.create({}), extras)
    assert environment.ground_truth == "42"
    assert extras == row
    extras["reward_spec"]["ground_truth"] = "changed"
    assert row == original
    with pytest.raises(AssertionError, match="reward_model field is required"):
        AIMEEnv(OmegaConf.create({}), {"reward_spec": row["reward_spec"]})


def test_contradictory_selected_gold_is_rejected():
    with pytest.raises(ValueError, match="contradictory reward channels"):
        reward_extras({"reward_spec": {"ground_truth": "42"}, "reward_model": {"ground_truth": "41"}})
