"""Per-token policy steps from the engine stamp to the training batch."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from rolloutengine.contracts import ModelTurn, RolloutData, RolloutStep, Transition
from taskcompendium.grading import GradeResult, Outcome, numeric_answer
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, Source, TaskSpec, TextMessage

from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.vllm.policy_steps import PolicyStepStamps
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutLease, RolloutTask
from skyrl_train.rollouts.task_projections import WholeTaskProjection, training_output
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.projections import WholeTrajectoryProjection
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics
from tests.cpu.util import example_dummy_config


def test_policy_step_stamps_follow_the_step_installed_before_each_engine_step():
    clock = [10.0]
    stamps = PolicyStepStamps(clock=lambda: clock[0])
    stamps.observe("a", 2, 10.0)
    clock[0] = 11.0
    stamps.install(1)
    clock[0] = 12.0
    stamps.observe("a", 3, 11.5)
    stamps.install(2)
    clock[0] = 13.0
    # Sampled before the second install and delivered after it: the step timestamp decides, not the arrival.
    stamps.observe("a", 1, 11.9)
    stamps.observe("a", 2, 12.5)
    stamps.observe("b", 4, 12.5)

    finished = stamps.finish("a")

    assert finished.dtype.name == "int32"
    assert finished.tolist() == [-1, -1, 1, 1, 1, 1, 2, 2]
    assert stamps.finish("a").tolist() == []
    stamps.discard(["b"])
    assert stamps.finish("b").tolist() == []


def test_projection_leaves_observation_tokens_between_stamped_turns_unsampled():
    def turn(prompt, response, step):
        return ModelTurn(
            {"role": "assistant", "content": "x"},
            prompt,
            response,
            None,
            "stop",
            metadata={"response_policy_steps": np.full(len(response), step, dtype=np.int32)},
        )

    rollout = RolloutData(
        task_id="task",
        messages=(),
        prompt_token_ids=(1, 2),
        response_token_ids=(3, 4, 5, 6, 99),
        loss_mask=(1, 1, 0, 1, 1),
        logprobs=None,
        grade=GradeResult(Outcome.GRADED, 1.0),
        stop_reason="stop",
        steps=(
            RolloutStep(turn((1, 2), (3, 4), 7), Transition(done=False), response_end=1, messages=()),
            RolloutStep(turn((1, 2, 3, 4, 5), (6, 99), 8), Transition(done=True), response_end=4, messages=()),
        ),
    )

    assert training_output(rollout).evidence.policy_steps.tolist() == [7, 7, -1, 8, 8]


class Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt):
        return [1, 2]

    def decode(self, token_ids, skip_special_tokens=False):
        return "12"


# One chat response per call: an attempt aborted by a weight sync, its continuation, then an unstamped response.
_SCRIPTED_RESPONSES = (
    ("1", [3, 4], [7, 7], "abort"),
    ("2", [5, 99], [8, 8], "stop"),
    ("12", [3, 99], None, "stop"),
)


class StampingEngine:
    def __init__(self):
        self.requests = []

    async def tokenize(self, payload):
        return {"tokens": [1, 2]}

    async def chat_completion(self, payload):
        self.requests.append(payload["json"])
        content, token_ids, policy_steps, finish_reason = _SCRIPTED_RESPONSES[len(self.requests) - 1]
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish_reason,
            "token_ids": token_ids,
            "logprobs": {"content": [{"logprob": -0.5} for _ in token_ids]},
        }
        if policy_steps is not None:
            choice["policy_steps"] = policy_steps
        return {
            "id": f"chatcmpl-{len(self.requests)}",
            "choices": [choice],
            "usage": {"prompt_tokens": 2, "completion_tokens": len(token_ids), "total_tokens": 2 + len(token_ids)},
        }


@dataclass
class Writer:
    groups: list[RolloutGroup] = field(default_factory=list)

    async def write_rollout(self, lease, group):
        self.groups.append(group)


def _task_request(repetition: int):
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        verifier=numeric_answer(12, tolerance_abs=0, tolerance_rel=0),
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    return {
        "prompts": [[{"role": "user", "content": "What is six plus six?"}]],
        "env_classes": ["taskcompendium"],
        "env_extras": [{"task_spec": task.model_dump_json(), "data_source": "arithmetic"}],
        "trajectory_ids": [TrajectoryID("arithmetic", repetition)],
        "sampling_params": None,
        "batch_metadata": None,
    }


@pytest.mark.asyncio
async def test_token_staleness_reaches_the_training_batch_through_stamped_rollouts():
    runner_config = OmegaConf.create(
        {
            "backend": "vllm",
            "max_turns": 1,
            "max_input_length": 100,
            "apply_overlong_filtering": False,
            "sampling_params": {
                "max_generate_length": 10,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": -1,
                "min_p": 0,
                "logprobs": 0,
            },
        }
    )
    client_config = OmegaConf.create(
        {
            "trainer": {"policy": {"model": {"path": "dummy-model"}}},
            "generator": {
                "backend": "vllm",
                "enable_http_endpoint": False,
                "http_endpoint_host": "127.0.0.1",
                "http_endpoint_port": 0,
                "weight_sync_pause_timeout_seconds": 30.0,
            },
        }
    )
    tokenizer = Tokenizer()
    engine = StampingEngine()
    worker = TaskRolloutWorker(
        runner_config,
        WholeTaskProjection(WholeTrajectoryProjection(runner_config, tokenizer)),
        DirectModelClient(InferenceEngineClient([engine], tokenizer, client_config)),
        {},
        command_timeout=5,
    )
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("a", policy_step=7, batch_id=9), {"uid": "a"}, _task_request(0)), writer)
        await worker.run_task(RolloutTask(RolloutLease("b", policy_step=8, batch_id=9), {"uid": "b"}, _task_request(1)), writer)
    finally:
        await worker.shutdown()

    # The continuation resumed the aborted attempt from its exact tokens, and the client joined both stamps.
    assert engine.requests[1]["continue_final_message"] is True
    assert engine.requests[1][EXACT_PROMPT_TOKEN_IDS_KEY] == [1, 2, 3, 4]
    stamped, unstamped = writer.groups
    assert stamped.trajectory_batch["response_ids"] == [[3, 4, 5, 99]]
    assert stamped.trajectory_batch["rollout_policy_steps"][0].tolist() == [7, 7, 8, 8]
    assert "rollout_policy_steps" not in unstamped.trajectory_batch

    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = example_dummy_config()
    trainer.context = SimpleNamespace(config=SimpleNamespace(batch_size=2, max_staleness_steps=2))
    trainer.global_step = 9
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = tokenizer
    trainer.pad_batch = lambda batch: batch
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer._training_metrics_enabled = False
    # The trainer's own postprocessing turns the per-response reward into a per-token reward on the last token.
    trainer.postprocess_trajectory_batch = lambda batch, uids: {
        **batch,
        "rewards": [
            [0.0] * (len(response) - 1) + [reward]
            for response, reward in zip(batch["response_ids"], batch["rewards"], strict=True)
        ],
    }
    trainer.select_trajectories = lambda batch, uids: (batch, uids)

    training_input = trainer.convert_rollout_groups_to_training_input([stamped, unstamped])

    # Stamped tokens charge their own step; the unstamped row charges its lease; padding is zero.
    assert training_input["rollout_staleness"].dtype == torch.int32
    assert training_input["rollout_staleness"].tolist() == [[2, 2, 1, 1], [1, 1, 0, 0]]
    assert {
        name.removeprefix("async/staleness_tokens_"): value
        for name, value in trainer.all_metrics.items()
        if name.startswith("async/staleness_tokens_")
    } == pytest.approx({"mean": 8 / 6, "max": 2, "ratio": 1.0, "stamped": 4 / 6, "spanning": 0.5})
    mismatch = mismatch_ratio_metrics(
        training_input["rollout_logprobs"],
        training_input["rollout_logprobs"],
        training_input["loss_mask"],
        training_input["rollout_staleness"],
    )
    assert [mismatch[f"policy/mismatch/staleness{bucket}/selected_tokens"] for bucket in (0, 1, 2)] == [0, 4, 2]

    # A stamp older than its lease cannot come from the engine that served the lease.
    stale_lease = RolloutGroup(stamped.trajectory_batch, "a", 8, {"uid": "a"})
    with pytest.raises(AssertionError, match="between its lease"):
        trainer.convert_rollout_groups_to_training_input([stale_lease, unstamped])
