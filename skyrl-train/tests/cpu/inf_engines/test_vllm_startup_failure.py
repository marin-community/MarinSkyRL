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


def test_independent_engine_retries_port_collisions(monkeypatch):
    pytest.importorskip("vllm")
    from skyrl_train.inference_engines.vllm import vllm_engine  # noqa: PLC0415

    engine = object()
    create_attempts = iter((RuntimeError("EADDRINUSE"), engine))
    sleeps = []

    def create_engine(*_args, **_kwargs):
        result = next(create_attempts)
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(vllm_engine, "_create_async_engine", create_engine)
    monkeypatch.setattr("random.uniform", lambda *_args: 2.0)
    monkeypatch.setattr(vllm_engine.time, "sleep", sleeps.append)

    result = vllm_engine._create_async_engine_with_port_collision_retries(object(), [])

    assert result is engine
    assert sleeps == [2.0, 15.0, 2.0]


def test_independent_engine_does_not_retry_other_failures(monkeypatch):
    pytest.importorskip("vllm")
    from skyrl_train.inference_engines.vllm import vllm_engine  # noqa: PLC0415

    attempts = 0

    def create_engine(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("model configuration is invalid")

    monkeypatch.setattr(vllm_engine, "_create_async_engine", create_engine)
    monkeypatch.setattr("random.uniform", lambda *_args: 2.0)
    monkeypatch.setattr(vllm_engine.time, "sleep", lambda *_args: None)

    with pytest.raises(RuntimeError, match="model configuration is invalid"):
        vllm_engine._create_async_engine_with_port_collision_retries(object(), [])

    assert attempts == 1
