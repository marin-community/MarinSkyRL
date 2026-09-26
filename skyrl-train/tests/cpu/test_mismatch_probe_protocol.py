import pytest
from omegaconf import DictConfig

import skyrl_gym

from skyrl_train.mismatch_probe.protocol import (
    probe_hash,
    request_seed,
    require_token_identity,
)


def test_probe_hash_is_stable_and_includes_order_masks_and_boundaries():
    record = ("a:0", [1, 2], [3, 4], [True, True], [True, False])
    assert probe_hash([record]) == probe_hash([record])
    assert probe_hash([record]) != probe_hash([("a:0", [1], [2, 3, 4], [True] * 3, [True, False, False])])
    assert probe_hash([record]) != probe_hash([("a:0", [1, 2], [3, 4], [True, True], [True, True])])
    assert probe_hash([record, ("b:0", [5], [6], [True], [True])]) != probe_hash(
        [("b:0", [5], [6], [True], [True]), record]
    )
    assert request_seed(7, "a", 0) == request_seed(7, "a", 0)
    assert request_seed(7, "a", 0) != request_seed(7, "a", 1)


def test_token_identity_mutation_is_caught_before_comparison():
    kwargs = dict(
        sample_id="a:0", expected_prompt=[1, 2], trainer_prompt=[1, 2], engine_response=[3, 4], trainer_response=[3, 4]
    )
    require_token_identity(**kwargs)
    with pytest.raises(ValueError, match="response IDs differ"):
        require_token_identity(**(kwargs | {"trainer_response": [3, 5]}))
    with pytest.raises(ValueError, match="prompt IDs differ"):
        require_token_identity(**(kwargs | {"trainer_prompt": [1, 8]}))


def test_fixture_reward_distinguishes_diverse_responses():
    env = skyrl_gym.make("mismatch_fixture", env_config=DictConfig({}), extras={})
    responses = ("blue square", "red circle", "green triangle", "yellow star")
    rewards = [env.step(response)["reward"] for response in responses]
    assert len(set(rewards)) == len(responses)
    assert env.step(responses[0])["reward"] == rewards[0]
    assert all(0 <= reward < 1 for reward in rewards)
