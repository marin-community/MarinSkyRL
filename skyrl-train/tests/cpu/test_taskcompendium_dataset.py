import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import msgspec
import pytest
from omegaconf import OmegaConf
from taskcompendium.execution import HarborTaskBinding, NoEnvironment, ShellSimEnvironment, ShellToolBinding
from taskcompendium.models import (
    AssistantFinal,
    Capability,
    Rendering,
    Source,
    StepSpecification,
    TaskMetadata,
    TaskRequirements,
    TaskSpec,
    TaskTroveVerifier,
)
from taskcompendium.serialization import to_json
from tasktrove_verify.spec import Mode

from skyrl_train.trajectory_runners.taskcompendium import (
    HARBOR_ENV_CLASS,
    NATIVE_CHAT_ENV_CLASS,
    NativeTaskCompendiumRunner,
    TaskCompendiumTaskDataset,
    TaskCompendiumTrajectoryRouter,
)
from skyrl_train.trajectory_runners.types import TrajectoryID


def _lowering(root: Path, name: str, *, harbor: bool = False) -> Path:
    task = root / name
    task.mkdir()
    requirements = TaskRequirements(capabilities=(Capability.FILESYSTEM,)) if harbor else TaskRequirements()
    specification = TaskSpec(
        id=name,
        steps=(
            StepSpecification(
                instructions="Reply with the word blue.",
                verifier=TaskTroveVerifier(Mode.EXACT, {"expected": ["blue"]}),
            ),
        ),
        requirements=requirements,
        resources=(),
        metadata=TaskMetadata(Source("test", "revision", name, "importer")),
    )
    binding = (
        HarborTaskBinding(ShellSimEnvironment(), (ShellToolBinding("shell", "shellsim"),))
        if harbor
        else HarborTaskBinding(NoEnvironment())
    )
    (task / "manifest.json").write_text("{}")
    (task / "specification.json").write_bytes(to_json(specification))
    (task / "renderings.json").write_bytes(msgspec.json.encode((Rendering("plain", AssistantFinal()),)))
    (task / "binding.json").write_bytes(msgspec.json.encode(binding))
    (task / "instruction.md").write_text("Reply with the word blue.")
    if harbor:
        execution = {
            "agent": {
                "import_path": "taskcompendium.harbor.agents:ShellToolAgent",
                "model_name": "stale-model",
                "kwargs": {"max_turns": 2},
            },
            "environment": {},
            "verifier": {},
        }
        (task / "reference-execution.json").write_text(json.dumps(execution))
    return task


def test_taskcompendium_dataset_routes_simple_chat_natively_and_binds_harbor_tasks(tmp_path):
    native = _lowering(tmp_path, "chat")
    harbor = _lowering(tmp_path, "tools", harbor=True)

    dataset = TaskCompendiumTaskDataset(
        [str(tmp_path)],
        api_base="http://policy:8000/v1",
        model_name="snowball",
    )

    assert [row["uid"] for row in dataset] == ["chat", "tools"]
    assert dataset[0] == {
        "uid": "chat",
        "prompt": [{"role": "user", "content": "Reply with the word blue."}],
        "env_class": NATIVE_CHAT_ENV_CLASS,
        "env_extras": {"task_dir": str(native)},
    }
    assert dataset[1]["env_class"] == HARBOR_ENV_CLASS
    assert dataset[1]["prompt"] == [{"role": "user", "content": str(harbor)}]
    execution = dataset[1]["env_extras"]["execution"]
    assert execution["agent"]["kwargs"]["api_base"] == "http://policy:8000/v1"
    assert execution["agent"]["model_name"] == "snowball"


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
    assert result["trajectory_ids"] == identities
    assert result["rollout_metrics"]["taskcompendium/native_chat_trajectories"] == 1.0
    assert result["rollout_metrics"]["taskcompendium/harbor_trajectories"] == 2.0
    assert native.requests[0]["prompts"] == [request["prompts"][1]]
    assert harbor.requests[0]["prompts"] == [request["prompts"][0], request["prompts"][2]]
