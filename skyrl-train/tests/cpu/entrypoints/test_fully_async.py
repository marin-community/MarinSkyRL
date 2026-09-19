import importlib
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from cloud.iris.rl_config_translation import RL_ENTRYPOINT_MODULES

from skyrl_train.entrypoints import fully_async


def test_trajectory_runner_uses_resolved_served_model_name(monkeypatch):
    cfg = OmegaConf.create(
        {
            "generator": {
                "enable_http_endpoint": True,
                "use_conversation_multi_turn": True,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 8000,
            },
            "environment": {"skyrl_gym": {}},
        }
    )
    inference_engine_client = MagicMock(model_name="served-policy")
    tokenizer = MagicMock()
    model_client = MagicMock()
    runner = MagicMock(custom_chat_template="template")
    create_model_client = MagicMock(return_value=model_client)
    create_runner = MagicMock(return_value=runner)
    monkeypatch.setattr(fully_async, "OpenAIHTTPModelClient", create_model_client)
    monkeypatch.setattr(fully_async, "SkyRLGymTrajectoryRunner", create_runner)

    result = fully_async.AsyncPPOExp.get_trajectory_runner(MagicMock(), cfg, tokenizer, inference_engine_client)

    assert result is runner
    create_model_client.assert_called_once_with(
        base_url="http://127.0.0.1:8000",
        model_name="served-policy",
        tokenizer=tokenizer,
    )
    assert create_runner.call_args.kwargs["model_client"] is model_client


def test_the_async_entrypoint_trains_inside_the_trainer_telemetry_lifecycle(monkeypatch):
    """The fully async loop runs under the trainer-role process owner and shuts down through the base run."""
    import skyrl_train.telemetry as training_telemetry
    from skyrl_train.entrypoints.fully_async import AsyncPPOExp

    monkeypatch.setenv("SKYRL_TELEMETRY_ENDPOINT", "http://finelog.test/v1/ingest")
    monkeypatch.setenv("SKYRL_RUN_ID", "async-entrypoint-test")
    monkeypatch.setenv("SKYRL_EXECUTION_UID", "test-attempt")
    monkeypatch.setattr(training_telemetry.telemetry, "configure", lambda **kwargs: None)
    seen: list[str] = []

    class Trainer:
        async def train(self):
            owner = training_telemetry._process_state.owner
            seen.append(owner._role if owner is not None else "unowned")

        async def shutdown(self):
            seen.append("shutdown")

    exp = AsyncPPOExp.__new__(AsyncPPOExp)
    exp.cfg = OmegaConf.create({"trainer": {"progress": {"mode": "off"}}})
    monkeypatch.setattr(exp, "_setup_trainer", lambda: Trainer())
    monkeypatch.setattr("skyrl_train.utils.progress.configure_progress", lambda progress: None)

    exp.run()

    assert seen == [training_telemetry.TRAINER_ROLE, "shutdown"]
    assert training_telemetry._process_state.owner is None


@pytest.mark.parametrize("module_name", sorted(RL_ENTRYPOINT_MODULES.values()))
def test_every_registered_entrypoint_runs_through_the_base_run(module_name):
    """An entrypoint that overrides run skips the trainer-role telemetry lifecycle the base run owns."""
    from skyrl_train.entrypoints.main_base import BasePPOExp

    module = importlib.import_module(module_name)
    experiments = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, BasePPOExp) and value.__module__ == module_name
    ]
    assert experiments, module_name
    for experiment in experiments:
        assert experiment.run is BasePPOExp.run, experiment
