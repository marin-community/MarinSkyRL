import pytest

from skyrl_train.inference_engines.vllm.utils import is_port_collision


@pytest.mark.parametrize(
    ("cause", "expected"),
    [(OSError(98, "Address already in use"), True), (None, False)],
)
def test_port_collision_retryability(cause, expected):
    failure = RuntimeError("Engine core initialization failed. See root cause above.")
    failure.__cause__ = cause

    assert is_port_collision(failure) is expected
