import json
from dataclasses import asdict, dataclass, field
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from taskcompendium.grader import verifyit_package
from taskcompendium.models import (
    AnswerType,
    ConversationInput,
    EnvironmentRequirements,
    PlainText,
    Source,
    TaskSpec,
    TextMessage,
)
from verifyit.spec import NumericSpec

from skyrl_train.group_admission import GroupAdvantageInvariant
from skyrl_train.inference_engines.chat_continuation import EXACT_PROMPT_TOKEN_IDS_KEY
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.vllm.policy_steps import PolicyStepStamps
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutLease, RolloutTask
from skyrl_train.rollouts.task_projections import WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.trajectory_runners.model_clients import DirectModelClient
from skyrl_train.trajectory_runners.types import TaskRequestBatch, TrajectoryID
from skyrl_train.utils.importance_ratio_diagnostics import mismatch_ratio_metrics
from tests.cpu.task_specs import lowered_task, task_input
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
    assert stamps.finish("a") is None
    stamps.discard(["b"])
    assert stamps.finish("b") is None


def test_policy_step_stamps_forget_requests_unclaimed_since_the_previous_sync():
    stamps = PolicyStepStamps(clock=lambda: 1.0)
    for request_id in ("streamed", "awaiting", "resident"):
        stamps.observe(request_id, 2, 0.5)

    stamps.forget_unclaimed({"resident"})
    assert stamps.finish("awaiting").tolist() == [-1, -1]
    stamps.forget_unclaimed({"resident"})

    assert stamps.finish("streamed") is None
    assert stamps.finish("resident").tolist() == [-1, -1]


class Tokenizer:
    eos_token_id = 99
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt):
        return [1, 2]

    def decode(self, token_ids, skip_special_tokens=False):
        return "12"


# One chat response per call: an attempt aborted by a weight sync, its continuation, an unstamped response, and a
# response sampled within one policy step.
_SCRIPTED_RESPONSES = (
    ("1", [3, 4], [7, 7], "abort"),
    ("2", [5, 99], [8, 8], "stop"),
    ("12", [3, 99], None, "stop"),
    ("12", [3, 99], [9, 9], "stop"),
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


def _task_request(repetition: int) -> TaskRequestBatch:
    task = TaskSpec(
        id="arithmetic",
        context=ConversationInput(events=(TextMessage(role="user", content="What is six plus six?"),)),
        environment_requirements=EnvironmentRequirements(),
        answer_type=AnswerType.NUMBER,
        answer_format=PlainText(),
        grader=verifyit_package(NumericSpec("12", tolerance_abs=0, tolerance_rel=0)).grader,
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    return TaskRequestBatch(
        rows=(task_input(lowered_task(task), data_source="arithmetic"),),
        trajectory_ids=(TrajectoryID("arithmetic", repetition),),
    )


def _worker(engine: StampingEngine) -> TaskRolloutWorker:
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
    return TaskRolloutWorker(
        runner_config,
        WholeTaskProjection(runner_config, tokenizer),
        DirectModelClient(InferenceEngineClient([engine], tokenizer, client_config)),
        {},
        shutdown_timeout=30,
    )


@pytest.mark.asyncio
async def test_stamped_turns_serialize_for_installed_agent_traces():
    worker = _worker(StampingEngine())
    try:
        (rollout,) = await worker.generate(_task_request(0))
    finally:
        await worker.shutdown()

    # Installed agents append every turn to their evidence trace as JSON.
    (step,) = rollout.steps
    assert json.loads(json.dumps(asdict(step.turn)))["metadata"]["response_policy_steps"] == [7, 7, 8, 8]


@pytest.mark.asyncio
async def test_token_staleness_reaches_the_training_batch_through_stamped_rollouts():
    engine = StampingEngine()
    worker = _worker(engine)
    writer = Writer()
    try:
        await worker.run_task(
            RolloutTask(RolloutLease("a", policy_step=7, batch_id=9), {"uid": "a"}, _task_request(0)), writer
        )
        await worker.run_task(
            RolloutTask(RolloutLease("b", policy_step=8, batch_id=9), {"uid": "b"}, _task_request(1)), writer
        )
        await worker.run_task(
            RolloutTask(RolloutLease("c", policy_step=9, batch_id=9), {"uid": "c"}, _task_request(2)), writer
        )
    finally:
        await worker.shutdown()

    # The continuation resumed the aborted attempt from its exact tokens, and the client joined both stamps.
    assert engine.requests[1]["continue_final_message"] is True
    assert engine.requests[1][EXACT_PROMPT_TOKEN_IDS_KEY] == [1, 2, 3, 4]
    stamped, unstamped, fresh = writer.groups
    assert stamped.trajectory_batch["response_ids"] == [[3, 4, 5, 99]]
    assert stamped.trajectory_batch["rollout_policy_steps"][0].tolist() == [7, 7, 8, 8]
    assert "rollout_policy_steps" not in unstamped.trajectory_batch

    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = example_dummy_config()
    trainer.context = SimpleNamespace(config=SimpleNamespace(batch_size=3, max_staleness_steps=2))
    trainer.global_step = 9
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = Tokenizer()
    trainer.pad_batch = lambda batch: batch
    trainer.group_advantage_invariant = GroupAdvantageInvariant.no_group_advantage(physical_group_size=1)
    trainer._training_metrics_enabled = True
    trainer.postprocess_trajectory_batch = lambda batch, uids: {
        **batch,
        "rewards": [
            [0.0] * (len(response) - 1) + [reward]
            for response, reward in zip(batch["response_ids"], batch["rewards"], strict=True)
        ],
    }
    trainer.select_trajectories = lambda batch, uids: (batch, uids)

    training_input = trainer.convert_rollout_groups_to_training_input([stamped, unstamped, fresh])

    # Stamped tokens charge their own step; the unstamped row charges its lease; padding is zero.
    assert training_input["rollout_staleness"].dtype == torch.int32
    assert training_input["rollout_staleness"].tolist() == [[2, 2, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]]
    assert {
        name.removeprefix("async/staleness_tokens_"): value
        for name, value in trainer.all_metrics.items()
        if name.startswith("async/staleness_tokens_")
    } == pytest.approx({"mean": 1.0, "max": 2, "ratio": 0.75, "stamped": 0.75, "spanning": 1 / 3})
    mismatch = mismatch_ratio_metrics(
        training_input["rollout_logprobs"],
        training_input["rollout_logprobs"],
        training_input["loss_mask"],
        training_input["rollout_staleness"],
    )
    assert [mismatch[f"policy/mismatch/staleness{bucket}/selected_tokens"] for bucket in (0, 1, 2)] == [2, 4, 2]

    # A stamp older than its lease cannot come from the engine that served the lease.
    stale_lease = RolloutGroup(stamped.trajectory_batch, "a", 8, {"uid": "a"})
    with pytest.raises(AssertionError, match="between its lease"):
        trainer.convert_rollout_groups_to_training_input([stale_lease, unstamped, fresh])
