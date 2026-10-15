import asyncio
import json

import httpx
import pytest
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.models.task.config import EnvironmentConfig
from rolloutengine.contracts import ModelTurn
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, MachineSpec, PackageSpec
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from skyrl_train.inference_engines.inference_engine_client_http_endpoint import create_app, set_global_state
from skyrl_train.rollouts.agent_backend import AgentInferenceBackend, TRIAL_HEADER
from skyrl_train.rollouts.agent_environment import BorrowedMachineEnvironment
from skyrl_train.rollouts.installed_agent import run_installed_agent


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "cancel"])
async def test_installed_agent_routes_http_and_retains_logs_after_execution(tmp_path, outcome):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "12": 1})))
    generating = asyncio.Event()

    async def tokenize(payload):
        return {"tokens": [2], "count": 1, "max_model_len": 32}

    async def complete(request):
        assert request.messages[-1]["content"] == "Solve the task."
        if outcome == "cancel":
            generating.set()
            await asyncio.Future()
        return ModelTurn({"role": "assistant", "content": "12"}, (2,), (1,), (-0.25,), "stop")

    backend = AgentInferenceBackend("model", 32, tokenize, tokenizer)
    set_global_state(backend, None)
    machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))

    class HttpAgent(BaseInstalledAgent):
        rollout_correlation_id = "trial"

        @staticmethod
        def name():
            return "http-fixture"

        async def install(self, environment):
            result = await environment.exec("echo installed > /logs/agent/setup.txt")
            assert result.return_code == 0

        async def run(self, instruction, environment, context):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=create_app()), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    headers={TRIAL_HEADER: self.rollout_correlation_id},
                    json={"model": "model", "messages": [{"role": "user", "content": instruction}]},
                )
            assert response.status_code == 200
            answer = response.json()["choices"][0]["message"]["content"]
            result = await environment.exec(f"echo {answer} > /logs/agent/answer.txt")
            assert result.return_code == 0
            context.n_output_tokens = response.json()["usage"]["completion_tokens"]
            if outcome == "failure":
                raise RuntimeError("Agent failed after its model response")

        def populate_context_post_run(self, context):
            context.metadata = {"pi_hydrated_trajectory": {"schema_version": 1, "turns": ["sample"]}}

    logs = tmp_path / "trial" / "agent"
    agent = HttpAgent(logs_dir=logs, version="fixture", model_name="hosted_vllm/model")
    environment = BorrowedMachineEnvironment(machine, tmp_path / "trial", "trial", EnvironmentConfig())
    try:
        execution = run_installed_agent(
            agent,
            environment,
            backend,
            complete,
            instruction="Solve the task.",
            setup_timeout=5,
            run_timeout=5,
            cleanup_timeout=5,
        )
        if outcome == "cancel":
            pending = asyncio.create_task(execution)
            await asyncio.wait_for(generating.wait(), timeout=5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        elif outcome == "failure":
            with pytest.raises(RuntimeError, match="Agent failed after its model response"):
                await execution
        else:
            assert await execution == {"schema_version": 1, "turns": ["sample"]}
        assert (logs / "setup.txt").read_text() == "installed\n"
        evidence = tmp_path / "trial" / "engine"
        records = [json.loads(line) for line in (evidence / "turns.jsonl").read_text().splitlines()]
        assert records[0]["request"]["messages"] == [{"role": "user", "content": "Solve the task."}]
        context = json.loads((evidence / "agent-context.json").read_text())
        if outcome == "cancel":
            assert len(records) == 1
            assert context["n_output_tokens"] is None
        else:
            assert (logs / "answer.txt").read_text() == "12\n"
            assert context["n_output_tokens"] == 1
            assert records[1]["turn"]["prompt_token_ids"] == [2]
            assert records[1]["turn"]["response_token_ids"] == [1]
            assert records[1]["turn"]["logprobs"] == [-0.25]
        with pytest.raises(ValueError, match="No active agent session"):
            await backend.chat_completion({"headers": {TRIAL_HEADER: "trial"}, "json": {}})
        result = await machine.run(Command(("cat", "/logs/agent/setup.txt")))
        assert result.stdout == b"installed\n"
    finally:
        await machine.close()
        set_global_state(None, None)
