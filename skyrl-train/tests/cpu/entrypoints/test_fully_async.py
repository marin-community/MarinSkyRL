import ast
import importlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from cloud.iris.rl_config_translation import RL_ENTRYPOINT_MODULES

import skyrl_train.telemetry as training_telemetry
from skyrl_train.entrypoints import fully_async
from skyrl_train.entrypoints.fully_async import AsyncPPOExp
from skyrl_train.entrypoints.main_base import BasePPOExp

SKYRL_TRAIN_ROOT = Path(__file__).resolve().parents[3]
# Every tree that can define a BasePPOExp subclass a user or a launcher runs.
EXPERIMENT_SOURCE_ROOTS = ("skyrl_train", "examples", "scripts", "integrations")


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
    module = importlib.import_module(module_name)
    experiments = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, BasePPOExp) and value.__module__ == module_name
    ]
    assert experiments, module_name
    for experiment in experiments:
        assert experiment.run is BasePPOExp.run, experiment


def _experiment_classes_overriding_run(source: Path) -> list[str]:
    """Name every BasePPOExp subclass in one source file that defines its own run()."""
    tree = ast.parse(source.read_text(), filename=str(source))
    experiments: set[str] = {"BasePPOExp"}
    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        base_names = {
            base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", None) for base in node.bases
        }
        if not base_names & experiments:
            continue
        experiments.add(node.name)
        if any(isinstance(body, (ast.FunctionDef, ast.AsyncFunctionDef)) and body.name == "run" for body in node.body):
            offenders.append(node.name)
    return offenders


def test_no_experiment_anywhere_in_the_tree_overrides_the_base_run():
    """An example or script that overrides run() also exports no telemetry; reading source covers
    trees whose imports CPU CI cannot satisfy."""
    offenders = {}
    for root in EXPERIMENT_SOURCE_ROOTS:
        for source in sorted((SKYRL_TRAIN_ROOT / root).rglob("*.py")):
            overriding = _experiment_classes_overriding_run(source)
            if overriding:
                offenders[str(source.relative_to(SKYRL_TRAIN_ROOT))] = overriding
    assert offenders == {}, offenders

def test_trajectory_runner_wraps_gym_with_harbor_for_terminal_bench_data(monkeypatch):
    cfg = OmegaConf.create(
        {
            "generator": {
                "enable_http_endpoint": True,
                "use_conversation_multi_turn": True,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 8000,
            },
            "environment": {"skyrl_gym": {}},
            "data": {"terminal_bench_data": ["tasks.parquet"]},
        }
    )
    inference_engine_client = MagicMock(model_name="served-policy")
    tokenizer = MagicMock()
    gym_runner = MagicMock(custom_chat_template="template")
    mixed_runner = MagicMock()
    monkeypatch.setattr(fully_async, "OpenAIHTTPModelClient", MagicMock())
    monkeypatch.setattr(fully_async, "SkyRLGymTrajectoryRunner", MagicMock(return_value=gym_runner))
    wrap = MagicMock(return_value=mixed_runner)
    monkeypatch.setattr(fully_async, "build_nemotron_ultra_trajectory_runner", wrap)

    result = fully_async.AsyncPPOExp.get_trajectory_runner(MagicMock(), cfg, tokenizer, inference_engine_client)

    assert result is mixed_runner
    wrap.assert_called_once_with(cfg, tokenizer, gym_runner)
