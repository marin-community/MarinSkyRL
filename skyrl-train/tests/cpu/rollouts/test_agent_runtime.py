from dataclasses import replace

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.trial.config import AgentConfig, EnvironmentConfig
from rolloutengine.contracts import ModelTurn, RolloutContractError
from rolloutengine.spec import LoweredTaskSpec, TaskRuntimeSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineSpec, PackageSpec
from shellbox.image import RegistryImage, image_source_key
from taskcompendium.models import ConversationInput, TextMessage, Source
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend, TRIAL_HEADER
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.agent_endpoint import AgentEndpoint
from skyrl_train.dataset.harbor_import import harbor_trial_task
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from tests.cpu.rollouts.engine_fakes import InferenceClient, Tokenizer as ProjectionTokenizer
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task, task_input


class UnavailableEnvironment(BaseEnvironment):
    """External container service fails before any agent command can run."""

    @staticmethod
    def type():
        return "docker"

    @property
    def capabilities(self):
        return EnvironmentCapabilities(mounted=True)

    def _validate_definition(self):
        pass

    async def start(self, force_build):
        raise RuntimeError("container service unavailable")

    async def stop(self, delete):
        pass

    async def upload_file(self, source_path, target_path):
        raise AssertionError("No transfers before startup")

    async def upload_dir(self, source_dir, target_dir):
        raise AssertionError("No transfers before startup")

    async def download_file(self, source_path, target_path):
        raise AssertionError("Logs use host mounts")

    async def download_dir(self, source_dir, target_dir):
        raise AssertionError("Logs use host mounts")

    async def exec(self, command, cwd=None, env=None, timeout_sec=None):
        raise AssertionError("No commands before startup")


@pytest.mark.asyncio
async def test_original_trial_mints_capability_for_container_setup_and_agent_run(tmp_path, monkeypatch, task_inputs):
    requested = []

    class Controller:
        def mint_endpoint_token(self, name, *, ttl):
            requested.append(ttl.to_seconds())
            return SimpleNamespace(
                capability_url="http://localhost/capability",
                expires_at=SimpleNamespace(epoch_ms=(1000 + ttl.to_seconds()) * 1000),
            )

    @asynccontextmanager
    async def endpoint(_port, *, request_timeout):
        yield AgentEndpoint("/trial", Controller())

    monkeypatch.setattr("skyrl_train.rollouts.agent_runtime.registered_agent_endpoint", endpoint)
    monkeypatch.setattr("skyrl_train.rollouts.agent_endpoint.time.time", lambda: 1000)
    package = tmp_path / "task"
    package.mkdir()
    (package / "instruction.md").write_text("Solve the task")
    (package / "task.toml").write_text('[environment]\ndocker_image = "python:3.12"\n')
    (package / "tests").mkdir()
    (package / "tests/test.sh").write_text("exit 1\n")
    task = harbor_trial_task(package, source=Source(dataset="fixture", revision="1", row="task", importer_revision="1"))
    session = (
        task_inputs[1]
        .rows[0]
        .lowered.session.model_copy(
            update={"attempt_timeout": 3000, "cleanup_timeout": 120, "total_turn_timeout": None, "max_turns": 999999}
        )
    )
    lowered = LoweredTaskSpec(
        task=task, runtime=TaskRuntimeSpec(task_machine=None, verifier_machine=None), session=session
    )
    settings = InstalledAgentSettings(
        agent=AgentConfig(
            name="mini-swe-agent",
            kwargs={"version": "2.1.0", "max_turns": 32},
            override_timeout_sec=1800,
            override_setup_timeout_sec=300,
        ),
        output_path=str(tmp_path / "published"),
        setup_timeout=300,
        run_timeout=1800,
        cleanup_timeout=120,
        service_startup_timeout=5,
        service_shutdown_timeout=5,
        environment_snapshots={image_source_key(RegistryImage("python:3.12")): "approved"},
    )
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0})))

    async def unexpected_request(_payload):
        raise AssertionError("The environment fails before inference")

    runtime = InstalledAgentRuntime(
        settings, AgentInferenceBackend("model", 32768, unexpected_request, tokenizer), 8192
    )
    harbor = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"max_retries": 0}}))
    harbor = replace(
        harbor,
        environment=EnvironmentConfig(import_path=f"{__name__}:UnavailableEnvironment"),
    )
    await runtime.startup()
    original = runtime.original_session(lowered, "trial", harbor, "train")
    try:
        with pytest.raises(RolloutContractError, match="exited before requesting"):
            await original.prepare()
        results = list((tmp_path / "published").glob("*/trials/*/result.json"))
        result = json.loads(results[0].read_text())
        assert result["exception_info"]["exception_message"] == "container service unavailable"
        assert result["config"]["agent"]["kwargs"]["max_turns"] == 32
        # The token is minted before container setup, whose allowance is 600 seconds.
        assert requested[0] > 600 + 300 + 1800 + 120
    finally:
        await original.close()
        await runtime.shutdown()


class HttpFixtureAgent(BaseInstalledAgent):
    rollout_correlation_id = "runtime-fixture"

    def __init__(self, *args, api_base, fail=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_base = api_base
        self.fail = fail

    @staticmethod
    def name():
        return "runtime-fixture"

    async def install(self, environment):
        result = await environment.exec("echo ready > /logs/agent/setup.txt")
        assert result.return_code == 0

    async def run(self, instruction, environment, context):
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.post(
                self.api_base + "/chat/completions",
                headers={TRIAL_HEADER: self.rollout_correlation_id},
                json={"model": "model", "messages": [{"role": "user", "content": instruction}]},
            )
        response.raise_for_status()
        context.n_output_tokens = response.json()["usage"]["completion_tokens"]
        if self.fail:
            raise RuntimeError("Fixture agent failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("path,fail", [("direct", False), ("direct", True), ("worker", False)])
async def test_native_runtime_routes_through_proxy_and_publishes_failed_attempts(
    tmp_path, monkeypatch, task_inputs, path, fail
):
    @asynccontextmanager
    async def local_endpoint(port, *, request_timeout):
        async def api_base(*, required_lifetime):
            return f"http://127.0.0.1:{port}/v1"

        yield SimpleNamespace(api_base=api_base)

    monkeypatch.setattr("skyrl_train.rollouts.agent_runtime.registered_agent_endpoint", local_endpoint)
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "12": 1})))

    async def tokenize(payload):
        return {"tokens": [2], "count": 1, "max_model_len": 32}

    async def complete(request):
        assert request.messages == ({"role": "user", "content": "Original BFCL instruction"},)
        return ModelTurn({"role": "assistant", "content": "12"}, (2,), (1,), (-0.25,), "stop")

    settings = InstalledAgentSettings(
        agent=AgentConfig(
            import_path=f"{__name__}:HttpFixtureAgent", model_name="hosted_vllm/checkpoint", kwargs={"fail": fail}
        ),
        output_path=str(tmp_path / "published"),
        setup_timeout=5,
        run_timeout=5,
        cleanup_timeout=5,
        service_startup_timeout=5,
        service_shutdown_timeout=5,
    )
    runtime = InstalledAgentRuntime(settings, AgentInferenceBackend("model", 32, tokenize, tokenizer), 8)
    worker_config, request = task_inputs
    task = request.rows[0].lowered.task
    task = task.model_copy(
        update={
            "context": ConversationInput(events=(TextMessage(role="user", content="Original BFCL instruction"),)),
            "tags": ("harbor",),
            "environment_requirements": SHELL_ENVIRONMENT,
        }
    )
    lowered = lowered_task(
        task,
        backend="shellsim",
        total_turn_timeout=5,
    )
    machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))
    await runtime.startup()
    try:
        if path == "worker":
            request = replace(request, rows=(task_input(lowered),))
            worker = TaskRolloutWorker(
                worker_config,
                WholeTaskProjection(worker_config, ProjectionTokenizer()),
                InferenceClient(),
                {"shellsim": ShellSimMachineFactory()},
                harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"max_retries": 0}})),
                installed_agents=runtime,
                shutdown_timeout=10,
            )
            try:
                batch = await worker.run(request)
                assert batch["unshaped_rewards"] == [1.0]
                assert batch["response_ids"] == [[3, 4]]
                assert [sum(rewards) for rewards in batch["rewards"]] == [1.0]
            finally:
                await worker.shutdown()
        elif fail:
            with pytest.raises(RuntimeError, match="Fixture agent failed"):
                await runtime.run(machine, complete, lowered=lowered, session_id="session")
        else:
            await runtime.run(machine, complete, lowered=lowered, session_id="session")
        attempts = list((tmp_path / "published").iterdir())
        assert len(attempts) == 1
        attempt = attempts[0]
        assert (attempt / "agent/setup.txt").read_text() == "ready\n"
        records = [json.loads(line) for line in (attempt / "engine/turns.jsonl").read_text().splitlines()]
        assert records[1]["turn"]["response_token_ids"] == ([3, 4] if path == "worker" else [1])
        identity = json.loads((attempt / "identity.json").read_text())
        assert identity["task_id"] == task.id
        if path == "direct":
            assert identity["session_id"] == "session"
        assert json.loads((attempt / "engine/agent-info.json").read_text())["model_info"]["name"] == "checkpoint"
    finally:
        await runtime.shutdown()
        await machine.close()
