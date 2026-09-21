from skyrl_train.inference_engines.vllm.utils import is_port_collision


def test_port_collision_is_retryable_through_exception_chain():
    collision = OSError(98, "Address already in use")
    wrapped = RuntimeError("Engine core initialization failed")
    wrapped.__cause__ = collision

    assert is_port_collision(wrapped)


def test_generic_engine_core_failure_is_not_retryable():
    failure = RuntimeError("Engine core initialization failed. See root cause above.")

    assert not is_port_collision(failure)
