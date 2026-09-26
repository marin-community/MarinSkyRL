import pytest

from skyrl_train.mismatch_probe.protocol import (
    chosen_logprobs_from_prompt_logprobs,
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


def test_prompt_logprob_extraction_requires_each_chosen_token():
    scored = chosen_logprobs_from_prompt_logprobs(
        prompt_lengths=[2, 1],
        response_ids=[[7, 8], [9]],
        prompt_logprobs=[[None, {2: -0.1}, {7: -1.0}, {8: -2.0}], [None, {9: -3.0}]],
    )
    assert scored == [[-1.0, -2.0], [-3.0]]
    with pytest.raises(ValueError, match="omitted frozen token"):
        chosen_logprobs_from_prompt_logprobs(
            prompt_lengths=[1], response_ids=[[9]], prompt_logprobs=[[None, {8: -3.0}]]
        )
