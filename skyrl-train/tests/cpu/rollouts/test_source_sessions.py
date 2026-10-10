from dataclasses import replace
import json
import shutil
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf
from skyrl_gym.verification import VerificationStatus
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import ExitReason, Result, ShellSimBuiltins, UnsupportedMachineSpec
from taskcompendium.models import Source
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.dataset.tasks import SourceTaskDataset, source_row_task
from skyrl_train.dataset.nemotron_ultra import NemotronTaskDataset
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from skyrl_gym.source_task import source_task
from skyrl_train.config.utils import get_default_config
from tests.cpu.task_specs import lowered_task, machine_runtime, session_spec
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.trajectory_runners.model_clients import DirectModelClient, ModelServerError
from skyrl_train.rollout_observability import observe_rollout_call
from tests.cpu.rollouts.engine_fakes import ConversationClient, InferenceClient, Tokenizer, Writer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "projection_type,expected_rewards",
    [(WholeTaskProjection, [[0.0, 0.0], [0.0, 1.0]]), (StepTaskProjection, [[0.0, 0.0], 1.0])],
)
@pytest.mark.parametrize("session", ["nupa", "reasoning_gym"])
async def test_corrupt_source_task_is_masked_without_losing_its_graded_peer(
    task_inputs, projection_type, expected_rewards, session
):
    config, original = task_inputs
    failed = source_task(
        original["prompts"][0],
        {"reward_spec": {"ground_truth": "not JSON"}, "reward_model": {"ground_truth": "not JSON"}},
        {},
        Source(dataset="corrupt", revision="1", row="0", importer_revision="1"),
    )
    request = {
        **original,
        "prompts": original["prompts"] * 2,
        "env_classes": [session, "taskcompendium"],
        "env_extras": [
            {"lowered_task_spec": lowered_task(failed, session).model_dump_json(), "teacher_route": "corrupt"},
            original["env_extras"][0],
        ],
        "trajectory_ids": [TrajectoryID("corrupt", 0), TrajectoryID("arithmetic", 0)],
    }
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        InferenceClient(),
        {},
        shutdown_timeout=30,
    )
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": failed.id}, request), writer)
    finally:
        await worker.shutdown()
    batch = writer.groups[0][1].trajectory_batch
    assert len(writer.groups) == 1
    assert batch["response_ids"] == [[3, 4], [3, 4]]
    assert batch["loss_masks"] == [[0, 0], [1, 1]]
    assert batch["exclude_from_baseline"] == [True, False]
    assert batch["rewards"] == expected_rewards
    assert batch["exception_types"] == ["VerifierRuntimeError", None]
    assert [result.status for result in batch["verification_results"]] == [
        VerificationStatus.ERROR,
        VerificationStatus.VERIFIED,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize("model_failure", [False, True])
@pytest.mark.parametrize(
    "alias_file,alias_metadata,identifier",
    [
        ("config.json", {"instance_id": "Project-MONAI__MONAI-123"}, "project-monai__monai-123"),
        (
            "test_info.json",
            {"github_repo": "python-pillow/Pillow", "base_commit": "3ba81234"},
            "python-pillow__Pillow-3ba81234",
        ),
    ],
)
async def test_mixed_nemotron_tasks_run_without_the_original_sources(
    tmp_path, task_inputs, projection_type, alias_file, alias_metadata, identifier, model_failure
):
    source = tmp_path / "source"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "instruction.md").write_text("Repair the terminal task.")
    (source / "task.toml").write_text(
        '[environment]\nworkdir = "/workspace"\nallow_internet = false\ndocker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
        '[verifier]\nenvironment_mode = "separate"\n'
    )
    (source / "tests" / alias_file).write_text(json.dumps(alias_metadata))
    (source / "tests/test.sh").write_text(
        "#!/bin/sh\necho 'tests/test_task.py::test_fix PASSED'\necho 0.75 > /logs/verifier/reward.txt\n"
    )
    rows = []
    for index, route in enumerate(("gym", "terminal_bench", "gym")):
        rows.append(
            {
                "prompt": [{"role": "user", "content": "What is six plus six?"}],
                "env_class": "gsm8k_multi_turn",
                "reward_spec": {"ground_truth": "12"},
                "teacher_route": "code" if route == "terminal_bench" else "math",
                "data_source": f"source-{index}",
                "extra_info": {
                    "nemotron_ultra": {
                        "blend": "rlvr1",
                        "agent": "swe" if route == "terminal_bench" else "arithmetic",
                        "route": route,
                        "terminal_bench_instance_id": identifier if route == "terminal_bench" else None,
                    }
                },
            }
        )
    input_path = tmp_path / "input.parquet"
    pq.write_table(pa.Table.from_pylist(rows), input_path)
    prepared = NemotronTaskDataset(
        [str(input_path)],
        Tokenizer(),
        100,
        environment_configs={
            **OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
            "session": session_spec().model_dump(exclude={"task_session"}),
        },
        terminal_bench_data=[str(source)],
        cache_dir=tmp_path / "tasks",
        num_workers=1,
    )
    shutil.rmtree(source)
    input_path.unlink()
    prompts, environments, extras, uids = zip(*[prepared[index] for index in range(len(prepared))], strict=True)

    class MixedClient:
        async def generate(self, request):
            if model_failure:
                raise ModelServerError("unavailable", "request-1", 503)
            continuation = request["chat_continuations"][0]
            prompt = [1, 2] if continuation is None else continuation["served_prefix_token_ids"] + [90, 91]
            output = await InferenceClient().generate(request)
            output["prompt_ids"] = [prompt]
            # Gym must retain its second turn despite the Harbor turn limit of one.
            output["assistant_messages"] = [
                {"role": "assistant", "content": "#### 13" if continuation is None else "#### 12"}
            ]
            return output

    class ImageFactory:
        async def create(self, spec):
            return await ShellSimMachineFactory().create(replace(spec, source=ShellSimBuiltins(), cpus=None))

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "max_turns": 1,
                    "override_cpus": 2,
                    "enable_reward_shaping": True,
                    "reward_shaper": "pass_ratio",
                    "reward_parser": "pytest",
                    "enable_token_reward_channel": False,
                    "enable_error_classification": True,
                    "default_error_treatment": "zero",
                    "max_retries": 0,
                }
            }
        )
    )
    config, request = task_inputs
    config.error_handling = {"enable_error_classification": True, "default_error_treatment": "mask"}
    request.update(
        prompts=list(prompts),
        env_classes=list(environments),
        env_extras=list(extras),
        trajectory_ids=[TrajectoryID(uid, 0) for uid in uids],
    )
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        MixedClient(),
        {"harbor": ImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": "mixed"}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    if model_failure:
        assert batch["exclude_from_baseline"] == [True, False, True]
        assert batch["error_treatments"] == ["mask", "zero", "mask"]
        assert batch["loss_masks"] == [[0], [0], [0]]
        expected_rows = [0, 1, 2]
    elif projection_type is WholeTaskProjection:
        assert batch["unshaped_rewards"] == [1.0, 0.75, 1.0]
        assert [sum(reward) for reward in batch["rewards"]] == [1.1, 1.0, 1.1]
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1], [1, 1], [1, 1, 0, 0, 1, 1]]
        expected_rows = [0, 1, 2]
    else:
        assert batch["unshaped_rewards"] == [0.0, 1.0, 0.75, 0.0, 1.0]
        assert [sum(reward) for reward in batch["rewards"]] == [0.1, 1.0, 1.0, 0.1, 1.0]
        assert batch["response_ids"] == [[3, 4]] * 5
        assert batch["loss_masks"] == [[1, 1]] * 5
        expected_rows = [0, 0, 1, 2, 2]
    assert batch["teacher_route_keys"] == [rows[index]["teacher_route"] for index in expected_rows]
    assert batch["data_sources"] == [rows[index]["data_source"] for index in expected_rows]
    assert [identity.instance_id for identity in batch["trajectory_ids"]] == [uids[index] for index in expected_rows]
    assert batch["rollout_metrics"]["nemotron_ultra/coverage/rlvr1/arithmetic"] == 2
    assert batch["rollout_metrics"]["nemotron_ultra/coverage/rlvr1/swe"] == 1
    assert "test_fix" not in json.dumps(prompts)


@pytest.mark.asyncio
async def test_source_tasks_run_without_the_original_dataset(tmp_path, task_inputs):
    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "prompt": [{"role": "user", "content": "What is six plus six?"}],
                    "env_class": "gsm8k",
                    "reward_spec": {"ground_truth": "12"},
                    "teacher_route": "math",
                    "data_source": "arithmetic",
                    "extra_info": {"grade": 3},
                }
            ]
        ),
        source,
    )
    prepared = SourceTaskDataset(
        [str(source)],
        Tokenizer(),
        100,
        environment_configs={
            **OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
            "session": session_spec().model_dump(exclude={"task_session"}),
        },
        num_workers=1,
    )
    source.unlink()
    prompt, env, extras, uid = prepared[0]
    config, _ = task_inputs
    client = ConversationClient(["#### 12"])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        client,
        {},
        shutdown_timeout=30,
    )
    request = {
        "prompts": [prompt],
        "env_classes": [env],
        "env_extras": [extras],
        "trajectory_ids": [TrajectoryID(uid, 0)],
        "sampling_params": None,
        "batch_metadata": None,
    }
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": uid}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["teacher_route_keys"] == ["math"]
    assert batch["data_sources"] == ["arithmetic"]
    assert extras["extra_info"] == {"grade": 3}
    assert "ground_truth" not in json.dumps(client.requests[0]["prompts"])
    assert batch["response_ids"] == [[3, 4]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "environment,extras,responses,rewards",
    [
        ("cat_count", {"extra_info": {"n": 2}}, ["cat cat"], [0.0, 1.0]),
        ("gsm8k", {"extra_info": None, "reward_spec": {"ground_truth": "12"}}, ["#### 12"], [0.0, 1.0]),
        ("gsm8k", {"reward_spec": {"ground_truth": "12"}}, ["#### 13"], [0.0, 0.0]),
        ("mcq", {"reward_model": {"ground_truth": "B"}}, [r"\boxed{B}"], [0.0, 1.0]),
        (
            "gsm8k_multi_turn",
            {"reward_spec": {"ground_truth": "12"}},
            ["#### 13", "#### 12"],
            [0.0, 0.1, 0.0, 0.0, 0.0, 1.0],
        ),
    ],
)
async def test_unified_gym_tasks_preserve_grading_and_turn_credit(task_inputs, environment, extras, responses, rewards):
    config, request = task_inputs
    task = source_row_task(
        {"prompt": request["prompts"][0], "env_class": environment, **extras},
        0,
        source_name="fixture",
        environment_configs={
            **OmegaConf.to_container(get_default_config().environment.task_sessions, resolve=True),
            "session": session_spec().model_dump(exclude={"task_session"}),
        },
    )
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = [environment]
    model = ConversationClient(responses)
    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        model,
        {},
        shutdown_timeout=30,
    )
    batch = await runner.run(request)
    assert batch["rewards"] == [rewards]
    expected_grade = float(responses[-1] == "#### 12") if environment == "gsm8k_multi_turn" else sum(rewards)
    assert batch["unshaped_rewards"] == [expected_grade]
    assert batch["verification_results"][0].score == expected_grade
    if environment == "cat_count":
        assert batch["verification_results"][0].passed is True
        assert batch["env_metrics"][0]["exact_n2"] == 1.0
    expected_mask = [1, 1] if len(responses) == 1 else [1, 1, 0, 0, 1, 1]
    assert batch["loss_masks"] == [expected_mask]
    assert batch["rollout_routed_experts"][0][:, 0, 0].tolist() == expected_mask
    assert len(batch["student_topk_indices"][0]) == len(expected_mask)
    assert all("ground_truth" not in str(item["prompts"]) for item in model.requests)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row_limits,session_config,turns",
    [
        ({"max_turns": 2}, {}, 2),
        ({"extra_info": {"max_turns": 2}}, {}, 2),
        ({"max_turns": None, "extra_info": {"max_turns": 2}}, {}, 2),
        ({"max_turns": 2, "extra_info": {"max_turns": 1}}, {}, 2),
        ({"max_turns": 2}, {"session": {"max_turns": 1}}, 1),
        ({}, {}, 1),
    ],
)
async def test_source_row_turn_limit_controls_correction_and_terminal_grade(
    task_inputs, row_limits, session_config, turns
):
    config, request = task_inputs
    task = source_row_task(
        {
            "prompt": request["prompts"][0],
            "env_class": "gsm8k_multi_turn",
            "reward_spec": {"ground_truth": "12"},
            **row_limits,
        },
        0,
        source_name="fixture",
        environment_configs={
            "session": session_spec(max_turns=1).model_dump(exclude={"task_session"}),
            "gsm8k_multi_turn": session_config,
        },
    )
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = ["gsm8k_multi_turn"]
    model = ConversationClient(["#### 13", "#### 12"])
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), model, {}, shutdown_timeout=30
    )
    batch = await worker.run(request)
    assert len(model.requests) == turns
    assert batch["unshaped_rewards"] == [float(turns == 2)]
    assert batch["verification_results"][0].passed is (turns == 2)
    assert sum(batch["rewards"][0]) == pytest.approx(0.2 / turns + float(turns == 2))


@pytest.mark.asyncio
async def test_materialized_turn_limit_precedes_source_row_metadata(task_inputs):
    config, request = task_inputs
    task = source_task(
        request["prompts"][0],
        {"reward_spec": {"ground_truth": "12"}, "max_turns": 1},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "gsm8k_multi_turn").model_dump_json()}]
    request["env_classes"] = ["gsm8k_multi_turn"]
    model = ConversationClient(["#### 13", "#### 12"])
    worker = TaskRolloutWorker(
        config, WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())), model, {}, shutdown_timeout=30
    )
    batch = await worker.run(request)
    assert len(model.requests) == 2
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["verification_results"][0].passed is True


@pytest.mark.asyncio
@pytest.mark.parametrize("enable_thinking", [True, False])
@pytest.mark.parametrize("reward_key", ["reward_spec", "reward_model"])
async def test_aime_rollout_preserves_length_reward_and_phase_metrics(
    task_inputs, delivered_telemetry, enable_thinking, reward_key
):
    config, request = task_inputs
    config.chat_template_kwargs = {"enable_thinking": enable_thinking}
    task = source_task(
        request["prompts"][0],
        extras={reward_key: {"ground_truth": "12"}},
        config={"length_penalty_weight": 1.0, "min_response_length": 0, "evaluation_token_budget": 1},
        source=Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "aime").model_dump_json()}]
    request["sampling_params"] = {"max_tokens": 4, "logprobs": 0}
    request["env_classes"] = ["aime"]
    engine = AsyncMock()
    engine.model_name = "fixture"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = r"Answer: \boxed{12}"

    async def tokenize(payload):
        thinking = payload["json"].get("chat_template_kwargs", {}).get("enable_thinking", True)
        return {"tokens": [1, 2, 20] if thinking else [1, 2]}

    engine.tokenize.side_effect = tokenize

    async def serve(payload):
        assert payload["json"]["max_completion_tokens"] == 4
        assert payload["json"]["chat_template_kwargs"]["enable_thinking"] is enable_thinking
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": r"Answer: \boxed{12}"},
                    "finish_reason": "stop",
                    "token_ids": [3, 4],
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                }
            ]
        }

    engine.chat_completion.side_effect = serve
    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        DirectModelClient(engine),
        {},
        max_verifier_workers=1,
        shutdown_timeout=30,
    )
    try:
        with observe_rollout_call(step=3, mode="async", enabled=True):
            batch = await runner.run(request)
    finally:
        await runner.shutdown()
    assert batch["rewards"][0] == pytest.approx([0.0, 0.5])
    assert batch["prompt_token_ids"] == [[1, 2, 20] if enable_thinking else [1, 2]]
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["loss_masks"] == [[1, 1]]
    assert batch["verification_results"][0].diagnostics["over_evaluation_budget"] is True
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
    phases = {
        row["attributes"]["phase"]: row["attributes"].get("parent")
        for row in delivered_telemetry.select("phase_duration_seconds", root="rollout_call", step="3")
    }
    assert phases == {
        "rollout_call": None,
        "rollout_collect": "rollout_call",
        "rollout_tokenize": "rollout_collect",
        "rollout_assemble": "rollout_call",
        "rollout_finalize": "rollout_call",
        "rollout_call_residual": "rollout_call",
    }
    waits = {row["attributes"]["wait"]: row["value"] for row in delivered_telemetry.select("rollout_waits", step="3")}
    assert waits["model_client_await"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "max_model_len,max_input_length,output_limit,expected_budgets",
    [
        (5, 100, None, [3]),
        (5, 100, 2, [2]),
        (2, 100, None, []),
        (None, 4, None, [10]),
        (None, 1, None, []),
        (100, 4, None, [10]),
        (100, 1, None, []),
    ],
)
async def test_context_limits_preserve_only_completed_gym_turns(
    task_inputs, projection_type, max_model_len, max_input_length, output_limit, expected_budgets
):
    config, request = task_inputs
    config.max_input_length = max_input_length
    config.engine_init_kwargs = {"max_model_len": max_model_len}
    config.sampling_params.logprobs = 0
    request["sampling_params"] = None if output_limit is None else {"max_tokens": output_limit}
    task = source_task(
        request["prompts"][0],
        {"reward_spec": {"ground_truth": "12"}},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [{"lowered_task_spec": lowered_task(task, "gsm8k_multi_turn").model_dump_json()}]
    request["env_classes"] = ["gsm8k_multi_turn"]
    engine = AsyncMock()
    engine.model_name = "fixture"
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "#### 13"
    budgets = []

    async def tokenize(payload):
        tokens = []
        for index, message in enumerate(payload["json"]["messages"]):
            if message["role"] == "assistant":
                tokens.extend([3, 4, 99] if message["content"] else [99])
            else:
                tokens.extend([1, 2] if index == 0 else [90, 91])
        return {"tokens": tokens}

    async def serve(payload):
        budgets.append(payload["json"]["max_completion_tokens"])
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "#### 13"},
                    "finish_reason": "stop",
                    "token_ids": [3, 4],
                    "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
                }
            ]
        }

    engine.tokenize.side_effect = tokenize
    engine.chat_completion.side_effect = serve
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(config, projection_type(projection), DirectModelClient(engine), {}, shutdown_timeout=30)
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert budgets == expected_budgets
    assert batch["stop_reasons"] == ["length"]
    assert batch["prompt_token_ids"] == [[1, 2]]
    if expected_budgets:
        assert batch["response_ids"] == [[3, 4]]
        assert batch["loss_masks"] == [[1, 1]]
        np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
        assert batch["unshaped_rewards"] == [0.0]
        assert batch["rewards"] == [[0.0, 0.1]]
        assert batch["evidence_messages"][0][-1] == {"role": "assistant", "content": "#### 13"}
    else:
        assert batch["response_ids"] == [[0]]
        assert batch["loss_masks"] == [[0]]
        np.testing.assert_allclose(batch["rollout_logprobs"], [[0.0]])
        assert batch["verification_results"][0].score is None
        assert batch["exclude_from_baseline"] == [True]


@pytest.mark.asyncio
async def test_lean_refinement_discards_the_failed_attempt(task_inputs, task_machine):
    config, request = task_inputs
    extras = {
        "extra_info": {
            "nemotron_ultra": {
                "route": "task_session",
                "agent": "math_formal_lean_refinement_agent",
                "record_json": json.dumps({"header": "", "formal_statement": "theorem equality : 1 = 1 := by sorry"}),
                "request_json": "{}",
            }
        }
    }
    task = source_task(
        request["prompts"][0],
        extras,
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    request["env_extras"] = [
        {"lowered_task_spec": lowered_task(task, "nemotron_ultra", backend="shellsim").model_dump_json()}
    ]
    request["env_classes"] = ["nemotron_ultra"]
    client = ConversationClient(["", "by rfl"])
    runner = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        client,
        {"shellsim": task_machine},
        shutdown_timeout=30,
    )
    batch = await runner.run(request)
    assert batch["response_ids"] == [[5, 6]]
    assert batch["loss_masks"] == [[1, 1]]
    assert batch["rewards"] == [[0.0, 1.0]]
    assert client.requests[1]["chat_continuations"] == [None]
    assert task_machine.closed


@pytest.fixture
def task_machine():
    """Supply compiler and interpreter I/O for the training-projection tests."""

    class Machine:
        closed = False
        output = "4"

        async def create(self, spec):
            self.output = spec.env.get("PYTHON_OUTPUT", "4")
            return self

        async def upload(self, source, target):
            pass

        async def run(self, command):
            data = json.loads(command.stdin) if command.stdin else {}
            if "proof" in data:
                output = {
                    "exit_code": 0,
                    "stdout": "",
                    "stderr": "",
                    "reason": "exited",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                }
            elif "code" in data:
                output = {
                    "exit_code": 0,
                    "stdout": self.output,
                    "stderr": "",
                    "reason": "exited",
                    "stdout_truncated": False,
                    "stderr_truncated": False,
                }
            else:
                output = {}
            return Result(0, json.dumps(output).encode(), b"", False, False, ExitReason.EXITED)

        async def close(self):
            self.closed = True

    return Machine()


@pytest.mark.asyncio
@pytest.mark.parametrize("per_agent", [False, True])
async def test_source_machine_selection_controls_tool_results(task_inputs, task_machine, per_agent):
    config, request = task_inputs
    environment = {
        "machine": {
            "requirements": {"environment_variables": {"PYTHON_OUTPUT": "4"}},
            "runtime": machine_runtime("shellsim").model_dump(),
        },
        "machines": {
            "ns_tools_simple_agent": {
                "requirements": {"environment_variables": {"PYTHON_OUTPUT": "7"}},
                "runtime": machine_runtime("shellsim").model_dump(),
            }
        }
        if per_agent
        else {},
    }
    expected = "7" if per_agent else "4"
    row = {
        "prompt": request["prompts"][0],
        "env_class": "nemotron_ultra",
        "extra_info": {
            "nemotron_ultra": {
                "route": "task_session",
                "agent": "ns_tools_simple_agent",
                "record_json": json.dumps({"expected_answer": expected}),
                "request_json": "{}",
            }
        },
    }
    task = source_row_task(
        row,
        0,
        source_name="source",
        environment_configs={
            "session": session_spec().model_dump(exclude={"task_session"}),
            "nemotron_ultra": environment,
        },
    )
    request["env_extras"] = [{"lowered_task_spec": task.model_dump_json()}]
    request["env_classes"] = ["nemotron_ultra"]
    message = {
        "role": "assistant",
        "tool_calls": [
            {"id": "python-1", "function": {"name": "stateful_python_code_exec", "arguments": '{"code":"2+2"}'}}
        ],
    }
    model = ConversationClient(["", expected], messages=[message, {"role": "assistant", "content": expected}])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        model,
        {"shellsim": task_machine},
        shutdown_timeout=30,
    )
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
    assert model.requests[1]["prompts"][0][-1] == {
        "role": "tool",
        "tool_call_id": "python-1",
        "content": expected,
    }
    assert batch["verification_results"][0].score == 1.0
    assert batch["exclude_from_baseline"] == [False]
    assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
    assert task_machine.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize(
    "session,operation,cause_type",
    [
        ("searchcode", "prepare", "RuntimeError"),
        ("nemotron_ultra", "prepare", "RuntimeError"),
        ("searchcode", "start", "UnsupportedMachineSpec"),
        ("searchcode", "start", "RuntimeError"),
    ],
)
async def test_python_startup_failure_is_masked_without_losing_its_graded_peer(
    task_inputs, task_machine, projection_type, session, operation, cause_type
):
    if operation == "start":
        startup_error = UnsupportedMachineSpec if cause_type == "UnsupportedMachineSpec" else RuntimeError
        task_machine.create = AsyncMock(side_effect=startup_error("The task provider did not start the machine"))
    task_machine.run = AsyncMock(
        return_value=Result(1, b"", b"ModuleNotFoundError: No module named 'IPython'", False, False, ExitReason.EXITED)
    )
    config, original = task_inputs
    config.error_handling = {"enable_error_classification": True}
    extras = (
        {"reward_spec": {"ground_truth": "12"}}
        if session == "searchcode"
        else {
            "extra_info": {
                "nemotron_ultra": {
                    "route": "task_session",
                    "agent": "ns_tools_simple_agent",
                    "record_json": json.dumps({"expected_answer": "12"}),
                    "request_json": "{}",
                }
            }
        }
    )
    failed = source_task(
        original["prompts"][0],
        extras,
        {},
        Source(dataset="python-startup", revision="1", row="0", importer_revision="1"),
    )
    request = {
        **original,
        "prompts": original["prompts"] * 2,
        "env_classes": [session, "taskcompendium"],
        "env_extras": [
            {
                "lowered_task_spec": lowered_task(failed, session, backend="shellsim").model_dump_json(),
                "teacher_route": "python-startup",
            },
            original["env_extras"][0],
        ],
        "trajectory_ids": [TrajectoryID(failed.id, 0), TrajectoryID("arithmetic", 0)],
    }
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())
    worker = TaskRolloutWorker(
        config, projection_type(projection), InferenceClient(), {"shellsim": task_machine}, shutdown_timeout=30
    )
    writer = Writer()
    try:
        await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": failed.id}, request), writer)
    finally:
        await worker.shutdown()
    batch = writer.groups[0][1].trajectory_batch
    assert batch["response_ids"] == [[0], [3, 4]]
    assert batch["loss_masks"] == [[0], [1, 1]]
    assert batch["exclude_from_baseline"] == [True, False]
    assert batch["exception_types"] == [
        "TaskMachineError" if operation == "start" and cause_type == "RuntimeError" else "VerifierRuntimeError",
        None,
    ]
    assert batch["error_treatments"] == ["mask", None]
    assert batch["unshaped_rewards"] == [0.0, 1.0]
    failed_result, peer_result = batch["verification_results"]
    assert failed_result.status is VerificationStatus.ERROR
    assert failed_result.diagnostics["operation"] == operation
    assert failed_result.diagnostics["cause_error_type"] == cause_type
    assert peer_result.status is VerificationStatus.VERIFIED
    assert peer_result.score == 1.0
    if operation == "prepare":
        assert task_machine.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_python_tool_observations_preserve_training_and_release_session(
    python_tool_task, task_inputs, task_machine, projection_type
):
    config, request = task_inputs
    _, message = python_tool_task
    model = ConversationClient(["", "4"], messages=[message, {"role": "assistant", "content": "4"}])
    projection = (
        WholeTrajectoryProjection(config, Tokenizer())
        if projection_type is WholeTaskProjection
        else StepWiseTrajectoryProjection(config, Tokenizer())
    )
    runner = TaskRolloutWorker(
        config,
        projection_type(projection),
        model,
        {"shellsim": task_machine},
        shutdown_timeout=30,
    )
    batch = await runner.run(request)
    assert task_machine.closed
    assert model.requests[1]["prompts"][0][-1] == {"role": "tool", "tool_call_id": "python-1", "content": "4"}
    if projection_type is WholeTaskProjection:
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
        assert batch["rewards"] == [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
        assert batch["exclude_from_baseline"] == [False]
    else:
        assert batch["loss_masks"] == [[1, 1], [1, 1]]
        assert batch["rewards"] == [[0.0, 0.0], [0.0, 1.0]]
        assert batch["exclude_from_baseline"] == [False, False]
