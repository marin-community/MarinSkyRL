from unittest.mock import Mock

import pytest

import skyrl_train.inference_engines.ray_wrapped_inference_engine as engine_module


def test_engine_startup_timeout_kills_all_engine_actors(monkeypatch):
    actors = [Mock(), Mock(), Mock()]
    refs = [object(), object(), object()]
    monkeypatch.setattr(engine_module.ray, "wait", lambda *args, **kwargs: (refs[:1], refs[1:]))
    kill = Mock()
    monkeypatch.setattr(engine_module.ray, "kill", kill)

    with pytest.raises(TimeoutError, match=r"inference engine startup timed out.*1, 2"):
        engine_module.wait_for_inference_engine_startup(refs, actors, timeout_seconds=60)

    assert [call.args[0] for call in kill.call_args_list] == actors


def test_engine_startup_wait_surfaces_actor_failure(monkeypatch):
    refs = [object(), object()]
    actors = [Mock(), Mock()]
    monkeypatch.setattr(engine_module.ray, "wait", lambda *args, **kwargs: (refs, []))
    get = Mock(side_effect=RuntimeError("engine initialization failed"))
    monkeypatch.setattr(engine_module.ray, "get", get)
    kill = Mock()
    monkeypatch.setattr(engine_module.ray, "kill", kill)

    with pytest.raises(RuntimeError, match="engine initialization failed"):
        engine_module.wait_for_inference_engine_startup(refs, actors, timeout_seconds=60)

    get.assert_called_once_with(refs)
    assert [call.args[0] for call in kill.call_args_list] == actors
