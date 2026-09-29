import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from omegaconf import OmegaConf
from taskcompendium.grading import exact_answer
from taskcompendium.lowering import HarborEnvironmentConfig, lower_to_harbor
from taskcompendium.models import AnswerType, Source, TaskRequirements, TaskSpec
from taskcompendium.resources import ResourceVisibility, TaskResource
from taskcompendium.submission import AnswerFormat, SubmissionConvention

from skyrl_train.entrypoints.taskcompendium import TaskCompendiumExp
from skyrl_train.rollouts.loader import PromptLoader, SeededPasses
from skyrl_train.trajectory_runners.taskcompendium import (
    HARBOR_ENV_CLASS,
    NATIVE_CHAT_ENV_CLASS,
    NativeTaskCompendiumRunner,
    TaskCompendiumHarborRunner,
    TaskCompendiumTaskDataset,
    TaskCompendiumTrajectoryRouter,
)
from skyrl_train.trajectory_runners.trajectory_processing import validate_trajectory_batch
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors


def _lowering(root: Path, name: str) -> Path:
    specification = TaskSpec(
        id=name,
        instructions="Reply with the word blue.",
        verifier=exact_answer("blue"),
        requirements=TaskRequirements(),
        source=Source(dataset="test", revision="revision", row=name, importer_revision="importer"),
        answer_type=AnswerType.TEXT,
        resources=(TaskResource(path="private/reference.txt", visibility=ResourceVisibility.VERIFIER, content="blue"),),
    )
    return lower_to_harbor(
        specification,
        SubmissionConvention(id="plain", answer_format=AnswerFormat.PLAIN),
        HarborEnvironmentConfig(),
        root / name,
    )


def test_taskcompendium_dataset_routes_simple_chat_natively(tmp_path):
    native = _lowering(tmp_path, "chat")

    dataset = TaskCompendiumTaskDataset(
        [str(tmp_path)],
        api_base="http://policy:8000/v1",
        model_name="snowball",
    )

    assert [row["uid"] for row in dataset] == ["chat"]
    assert dataset[0] == {
        "uid": "chat",
        "prompt": [{"role": "user", "content": "Reply with the word blue.\n\nGive your answer as plain text.\n"}],
        "env_class": NATIVE_CHAT_ENV_CLASS,
        "env_extras": {"task_dir": str(native)},
    }


def test_taskcompendium_dataset_supplies_distinct_uids_to_prompt_loader(tmp_path):
    _lowering(tmp_path, "chat-a")
    _lowering(tmp_path, "chat-b")
    dataset = TaskCompendiumTaskDataset([str(tmp_path)], api_base="http://policy:8000/v1", model_name="snowball")
    loader = PromptLoader(dataset, SeededPasses(len(dataset), seed=17, shuffle=False), batch_size=2)

    first = loader.next_prompt(set())
    assert first is not None
    second = loader.next_prompt({first["uid"]})

    assert first["uid"] == "chat-a"
    assert second is not None
    assert second["uid"] == "chat-b"


def test_taskcompendium_requires_a_serving_policy_endpoint():
    experiment = object.__new__(TaskCompendiumExp)
    experiment.cfg = OmegaConf.create(
        {
            "terminal_bench_config": {"agent_api_base": None},
            "generator": {"enable_http_endpoint": False, "http_endpoint_host": "127.0.0.1", "http_endpoint_port": 8000},
        }
    )

    with pytest.raises(ValueError, match="generator.enable_http_endpoint=true"):
        experiment._api_base()

    experiment.cfg.generator.enable_http_endpoint = True
    assert experiment._api_base() == "http://127.0.0.1:8000/v1"


@pytest.mark.asyncio
async def test_native_runner_preserves_engine_tokens_logprobs_and_grades_response(tmp_path):
    task = _lowering(tmp_path, "chat")
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [10, 11, 12]
    model_client = AsyncMock()
    model_client.generate.return_value = {
        "responses": ["blue"],
        "response_ids": [[20, 21]],
        "stop_reasons": ["stop"],
        "response_logprobs": [[-0.1, -0.2]],
        "prompt_logprobs": None,
        "token_provenance": "engine",
    }
    cfg = OmegaConf.create(
        {
            "sampling_params": {"temperature": 1.0},
            "chat_template": None,
            "chat_template_kwargs": {},
            "trajectory_reward_shaping": None,
        }
    )
    runner = NativeTaskCompendiumRunner(cfg, tokenizer, model_client)
    identity = TrajectoryID("chat", 0)
    request = {
        "prompts": [[{"role": "user", "content": "Reply with the word blue."}]],
        "env_classes": [NATIVE_CHAT_ENV_CLASS],
        "env_extras": [{"task_dir": str(task)}],
        "sampling_params": {"temperature": 0.5, "logprobs": 1},
        "trajectory_ids": [identity],
        "batch_metadata": None,
    }

    result = await runner.run(request)

    assert result["prompt_token_ids"] == [[10, 11, 12]]
    assert result["response_ids"] == [[20, 21]]
    assert result["rollout_logprobs"] == [[-0.1, -0.2]]
    assert result["loss_masks"] == [[1, 1]]
    assert result["rewards"] == [1.0]
    assert result["trajectory_ids"] == [identity]
    inference_request = model_client.generate.await_args.args[0]
    assert inference_request["prompt_token_ids"] == [[10, 11, 12]]
    assert inference_request["prompts"] is None


class StubRunner:
    def __init__(self, reward: float, *, logprobs: bool):
        self.reward = reward
        self.logprobs = logprobs
        self.requests = []

    async def startup(self):
        pass

    async def shutdown(self):
        pass

    async def start_eval_session(self, **kwargs):
        pass

    async def stop_eval_session(self):
        pass

    async def run(self, request, disable_tqdm=False):
        self.requests.append(request)
        size = len(request["prompts"])
        return {
            "prompt_token_ids": [[int(self.reward)]] * size,
            "response_ids": [[int(self.reward)]] * size,
            "rewards": [self.reward] * size,
            "unshaped_rewards": [self.reward] * size,
            "loss_masks": [[1]] * size,
            "stop_reasons": ["stop"] * size,
            "exception_types": [None] * size,
            "error_treatments": [None] * size,
            "rollout_metrics": {},
            "rollout_logprobs": ([[-0.1]] * size if self.logprobs else None),
            "trajectory_ids": list(request["trajectory_ids"]),
        }


@pytest.mark.asyncio
async def test_router_splits_mixed_batches_and_restores_order():
    native = StubRunner(1.0, logprobs=True)
    harbor = StubRunner(2.0, logprobs=False)
    router = TaskCompendiumTrajectoryRouter(
        native_runner=native,
        harbor_runner=harbor,
        require_rollout_logprobs=False,
        tis_lcs_alert_threshold=0.005,
    )
    identities = [TrajectoryID(str(index), 0) for index in range(3)]
    request = {
        "prompts": [[{"role": "user", "content": str(index)}] for index in range(3)],
        "env_classes": [HARBOR_ENV_CLASS, NATIVE_CHAT_ENV_CLASS, HARBOR_ENV_CLASS],
        "env_extras": [{"task_dir": str(index)} for index in range(3)],
        "sampling_params": {},
        "trajectory_ids": identities,
        "batch_metadata": SimpleNamespace(training_phase="train"),
    }

    result = await router.run(request)

    assert result["response_ids"] == [[2], [1], [2]]
    assert result["rewards"] == [2.0, 1.0, 2.0]
    assert result["rollout_logprobs"] is None
    assert result["trajectory_ids"] == identities
    assert result["rollout_metrics"]["taskcompendium/native_chat_trajectories"] == 1.0
    assert result["rollout_metrics"]["taskcompendium/harbor_trajectories"] == 2.0
    assert native.requests[0]["prompts"] == [request["prompts"][1]]
    assert harbor.requests[0]["prompts"] == [request["prompts"][0], request["prompts"][2]]


@pytest.mark.asyncio
async def test_mixed_batch_uses_live_scripted_policy_endpoint_and_produces_trainable_actions(tmp_path):
    pytest.importorskip("harbor")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from taskcompendium.importers.nemo_workplace import load_fixture
    from transformers import PreTrainedTokenizerFast

    _lowering(tmp_path, "chat")
    specification, convention, binding = load_fixture()
    lower_to_harbor(specification, convention, binding, tmp_path / "workplace")
    source_row = json.loads(
        next(resource.content for resource in specification.resources if resource.path == "source-row.json")
    )
    gold = source_row["ground_truth"][0]
    requests = []

    class Endpoint(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(payload)
            message = (
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": gold["name"], "arguments": gold["arguments"]},
                        }
                    ],
                }
                if len(requests) == 1
                else {"role": "assistant", "content": "Done."}
            )
            body = json.dumps({"choices": [{"message": message}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Endpoint)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        dataset = TaskCompendiumTaskDataset(
            [str(tmp_path / "chat"), str(tmp_path / "workplace")],
            api_base=f"http://127.0.0.1:{server.server_port}/v1",
            model_name="fixture",
        )
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=backend, unk_token="[UNK]", eos_token="[EOS]", pad_token="[UNK]"
        )
        tokenizer.chat_template = (
            "{% for message in messages %}{{ message['role'] }}: {{ message['content'] or '' }}"
            "{% if message.get('tool_calls') %}{{ message['tool_calls'] }}{% endif %} [EOS] {% endfor %}"
            "{% if add_generation_prompt %}assistant: {% endif %}"
        )
        client = AsyncMock()
        client.generate.return_value = {
            "responses": ["blue"],
            "response_ids": [[20, 21]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.1, -0.2]],
        }
        cfg = OmegaConf.create(
            {"sampling_params": {"temperature": 1.0}, "chat_template": None, "chat_template_kwargs": {}}
        )
        router = TaskCompendiumTrajectoryRouter(
            native_runner=NativeTaskCompendiumRunner(cfg, tokenizer, client),
            harbor_runner=TaskCompendiumHarborRunner(
                tokenizer, tmp_path / "trials", concurrency=1, max_turns=3, timeout=30
            ),
            require_rollout_logprobs=False,
            tis_lcs_alert_threshold=0.005,
        )
        batch = {
            "prompts": [row["prompt"] for row in dataset],
            "env_classes": [row["env_class"] for row in dataset],
            "env_extras": [row["env_extras"] for row in dataset],
            "sampling_params": {"temperature": 1.0},
            "trajectory_ids": [TrajectoryID(row["uid"], 0) for row in dataset],
            "batch_metadata": None,
        }
        await router.startup()
        try:
            result = await router.run(batch)
        finally:
            await router.shutdown()
    finally:
        server.shutdown()
        server.server_close()
        thread.join()

    assert result["rewards"] == [1.0, 1.0]
    validate_trajectory_batch(2, result)
    assert result["rollout_logprobs"] is None
    assert all(
        tokens and len(tokens) == len(mask) and any(mask)
        for tokens, mask in zip(result["response_ids"], result["loss_masks"], strict=True)
    )
    assert 0 in result["loss_masks"][1]
    tensor_batch = convert_prompts_responses_to_batch_tensors(
        tokenizer,
        result["prompt_token_ids"],
        result["response_ids"],
        [
            [0.0] * (len(tokens) - 1) + [reward]
            for tokens, reward in zip(result["response_ids"], result["rewards"], strict=True)
        ],
        result["loss_masks"],
    )
    assert tensor_batch[0].shape[0] == tensor_batch[4].shape[0] == 2
    assert tensor_batch[5] is None
    assert len(requests) == 2
    assert len(requests[0]["tools"]) == 27
    assert "ground_truth" not in json.dumps(requests)
    traces = list((tmp_path / "trials").rglob("result.json"))
    assert len(traces) == 1
    actions = json.loads(traces[0].read_text())["agent_result"]["metadata"]["tools"]
    assert [action["call_id"] for action in actions] == ["call-1"]
