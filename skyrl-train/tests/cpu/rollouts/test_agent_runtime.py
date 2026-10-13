import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx
import pytest
from omegaconf import OmegaConf
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.trial.config import AgentConfig
from rolloutengine.contracts import ModelTurn
from rolloutengine.spec import LoweredTaskSpec
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import MachineSpec, PackageSpec
from taskcompendium.models import ConversationInput, TextMessage
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.rollouts.agent_backend import AgentInferenceBackend, TRIAL_HEADER
from skyrl_train.rollouts.agent_runtime import InstalledAgentRuntime, InstalledAgentSettings
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from tests.cpu.rollouts.engine_fakes import InferenceClient, Tokenizer as ProjectionTokenizer
from tests.cpu.task_specs import SHELL_ENVIRONMENT, lowered_task


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
    async def local_endpoint(port):
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
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
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
            request["env_extras"] = [{"lowered_task_spec": lowered.model_dump_json()}]
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
