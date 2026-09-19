import pytest

from skyrl_train.fully_async_trainer import should_publish_policy_weights


def test_delayed_publication_keeps_evaluation_and_terminal_weights_current():
    assert [should_publish_policy_weights(step, 4, 8, 5) for step in range(1, 9)] == [
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
    ]
    assert all(should_publish_policy_weights(step, 1, 8, -1) for step in range(1, 9))


def test_weight_publication_rejects_nonpositive_interval():
    with pytest.raises(ValueError, match="positive"):
        should_publish_policy_weights(1, 0, 8, -1)
