import asyncio
import json

import httpx
import numpy as np
import pytest
from harbor.agents.installed.base import BaseInstalledAgent
from harbor.agents.installed.pi import Pi
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
from skyrl_train.rollouts.installed_agent import retain_pi_sandbox_diagnostics, run_installed_agent


@pytest.mark.asyncio
async def test_failed_pi_preserves_opt_in_sandbox_diagnostics(tmp_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0})))

    async def tokenize(payload):
        return {"tokens": [0], "count": 1, "max_model_len": 32}

    backend = AgentInferenceBackend("model", 32, tokenize, tokenizer)
    machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))

    class FailingPi(Pi):
        async def setup(self, environment):
            result = await environment.exec(
                'echo \'{"source":"budget","phase":"registered","context_window":32}\' '
                "> /logs/agent/pi-sandbox-diagnostics.jsonl"
            )
            assert result.return_code == 0

        async def run(self, instruction, environment, context):
            raise ValueError("No collated Pi trajectory")

    trial = tmp_path / "trial"
    agent = FailingPi(
        logs_dir=trial / "agent",
        model_name="hosted_vllm/model",
        api_base="http://localhost/v1",
        thinking_format="chat-template",
        collect_rollout_details=True,
        verbose_sandbox_diagnostics=True,
    )
    environment = BorrowedMachineEnvironment(machine, trial, "trial", EnvironmentConfig())
    try:
        with pytest.raises(ValueError):
            await run_installed_agent(
                agent,
                environment,
                backend,
                lambda request: asyncio.Future(),
                instruction="task",
                setup_timeout=5,
                run_timeout=5,
                cleanup_timeout=5,
            )
        diagnostic_path = trial / "engine" / "pi-sandbox-diagnostics.jsonl"
        assert json.loads(diagnostic_path.read_text()) == {
            "source": "budget",
            "phase": "registered",
            "context_window": 32,
        }
    finally:
        await machine.close()


def test_pi_sandbox_diagnostics_accept_only_fixed_safe_counters(tmp_path):
    source = tmp_path / "sandbox.jsonl"
    target = tmp_path / "published.jsonl"
    events = [
        {
            "source": "budget",
            "phase": "request",
            "attempt": 1,
            "context_window": 2048,
            "configured_output_tokens": 256,
            "message_count": 2,
            "tool_count": 4,
        },
        {"source": "budget", "phase": "tokenize_completed", "prompt_tokens": 2200},
        {
            "source": "budget",
            "phase": "decision",
            "attempt": 1,
            "original_prompt_tokens": 2200,
            "fitted_prompt_tokens": 2200,
            "prompt_budget": 1792,
            "available_output_tokens": 1,
            "output_tokens": 1,
            "tool_outputs_fitted": False,
        },
        {
            "source": "recorder",
            "phase": "upstream_status",
            "http_status": 200,
            "max_completion_tokens": 1,
            "streaming": True,
        },
        {
            "source": "recorder",
            "phase": "literal_evidence",
            "http_status": 200,
            "prompt_token_count": 1484,
            "completion_token_count": 1,
            "logprob_count": None,
            "has_assistant": True,
            "tool_call_count": 0,
        },
        {
            "source": "recorder",
            "phase": "capture_failed",
            "turn_count": 1,
            "missing_prompt_ids": False,
            "missing_completion_ids": False,
            "missing_logprobs": True,
            "missing_assistant": False,
        },
    ]
    source.write_text("\n".join(json.dumps(record) for record in events) + "\n")
    retain_pi_sandbox_diagnostics(source, target)
    assert [json.loads(line) for line in target.read_text().splitlines()] == events
    for invalid in (
        {**events[0], "url": "private-url"},
        {**events[0], "context_window": True},
        {**events[0], "phase": "unknown"},
        {**events[0], "context_window": -1},
    ):
        source.write_text(json.dumps(invalid) + "\n")
        with pytest.raises(ValueError, match="Pi sandbox diagnostic"):
            retain_pi_sandbox_diagnostics(source, target)
    assert [json.loads(line) for line in target.read_text().splitlines()] == events


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
        return ModelTurn(
            {"role": "assistant", "content": "12"},
            (2,),
            (1,),
            (-0.25,),
            "stop",
            metadata={"routed_experts": np.array([[[3, 7]]], dtype=np.int32)},
        )

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
        diagnostics = [json.loads(line) for line in (evidence / "diagnostics.jsonl").read_text().splitlines()]
        phases = [item["phase"] for item in diagnostics]
        assert "agent:setup_complete" in phases
        assert "agent:run_start" in phases
        assert "model_request:start" in phases
        assert ("model_request:failed" if outcome == "cancel" else "model_request:completed") in phases
        assert "agent:download_complete" in phases
        assert any(phase.startswith("machine_run:") for phase in phases)
        assert not any("command" in item or "url" in item for item in diagnostics)
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
            assert records[1]["turn"]["metadata"]["routed_experts"] == [[[3, 7]]]
        with pytest.raises(ValueError, match="No active agent session"):
            await backend.chat_completion({"headers": {TRIAL_HEADER: "trial"}, "json": {}})
        result = await machine.run(Command(("cat", "/logs/agent/setup.txt")))
        assert result.stdout == b"installed\n"
    finally:
        await machine.close()
        set_global_state(None, None)


@pytest.mark.asyncio
async def test_installed_agent_timeout_retains_diagnostics_when_machine_closes(tmp_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0})))

    async def tokenize(payload):
        return {"tokens": [0], "count": 1, "max_model_len": 32}

    backend = AgentInferenceBackend("model", 32, tokenize, tokenizer)
    machine = await ShellSimMachineFactory().create(MachineSpec(PackageSpec()))

    class StalledAgent(BaseInstalledAgent):
        rollout_correlation_id = "trial"

        @staticmethod
        def name():
            return "stalled-fixture"

        async def install(self, environment):
            pass

        async def run(self, instruction, environment, context):
            try:
                await asyncio.Future()
            finally:
                await machine.close()

    trial = tmp_path / "trial"
    agent = StalledAgent(logs_dir=trial / "agent", version="fixture", model_name="hosted_vllm/model")
    environment = BorrowedMachineEnvironment(machine, trial, "trial", EnvironmentConfig())
    try:
        with pytest.raises(TimeoutError) as failure:
            await run_installed_agent(
                agent,
                environment,
                backend,
                lambda request: asyncio.Future(),
                instruction="private task text",
                setup_timeout=5,
                run_timeout=0.05,
                cleanup_timeout=5,
            )
        assert failure.value.__cause__ is not None
        assert "Machine is closed" in str(failure.value.__cause__)
        diagnostics = [json.loads(line) for line in (trial / "engine" / "diagnostics.jsonl").read_text().splitlines()]
        assert any(item["phase"] == "agent:failed" and item["error_type"] == "TimeoutError" for item in diagnostics)
        assert diagnostics[-1]["phase"] == "agent:download_failed"
        assert "private task text" not in (trial / "engine" / "diagnostics.jsonl").read_text()
    finally:
        await machine.close()
