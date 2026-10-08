import asyncio
from dataclasses import replace
import json
import shutil

import numpy as np
import pytest
from omegaconf import OmegaConf
from skyrl_gym.verification import VerificationStatus
from shellbox.backends.shellsim.machine import ShellSimMachineFactory
from shellbox.machine import Command, ShellSimBuiltins, NetworkPolicy
from taskcompendium.shell_verifier import (
    ArtifactKind,
    FileReward,
    MissingArtifactPolicy,
    RewardFile,
    RewardFileFormat,
    ShellVerifierSpec,
    VerifierArtifact,
)
from taskcompendium.models import AnswerType, ConversationInput, EnvironmentRequirements, TextMessage, VerifierSpec
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.dataset.tasks import TaskDataset
from skyrl_train.dataset.harbor import HarborTaskDataset
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection, WholeTrajectoryProjection
from skyrl_train.rollouts.buffer import RolloutLease, RolloutTask
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.rollouts.harbor_tasks import HarborTaskSettings
from rolloutengine.spec import LoweredTaskSpec
from tests.cpu.task_specs import lowered_task, session_spec
from skyrl_train.trajectory_runners.types import BatchMetadata, TokenProvenance, TrajectoryID
from skyrl_train.trajectory_runners.model_clients import ModelServerError
from tests.cpu.rollouts.engine_fakes import ConversationClient, FixtureImageFactory, InferenceClient, Tokenizer, Writer


@pytest.mark.asyncio
async def test_mixed_harbor_and_native_tasks_keep_separate_providers_and_machine_specs(task_inputs):
    config, original = task_inputs
    lowered = LoweredTaskSpec.model_validate_json(original["env_extras"][0]["lowered_task_spec"])
    native = lowered.task.model_copy(
        update={"environment_requirements": EnvironmentRequirements(docker_image="fixture@sha256:" + "a" * 64)}
    )
    harbor = native.model_copy(update={"id": "harbor", "tags": ("harbor",)})
    machines = {"docker": [], "harbor": []}

    class Factory:
        def __init__(self, provider):
            self.provider = provider

        async def create(self, spec):
            machines[self.provider].append(spec)
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), cpus=None, workdir="/workspace")
            )

    settings = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"override_cpus": 2, "max_retries": 0}}))
    request = {
        **original,
        "prompts": original["prompts"] * 2,
        "env_classes": ["taskcompendium"] * 2,
        "env_extras": [
            {"lowered_task_spec": lowered_task(task, backend="docker").model_dump_json()} for task in (native, harbor)
        ],
        "trajectory_ids": [TrajectoryID(task.id, 0) for task in (native, harbor)],
    }
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {provider: Factory(provider) for provider in machines},
        harbor=settings,
        shutdown_timeout=30,
    )
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
    assert batch["unshaped_rewards"] == [1.0, 1.0]
    assert [sum(reward) for reward in batch["rewards"]] == [1.0, 1.0]
    assert batch["exclude_from_baseline"] == [False, False]
    assert len(machines["docker"]) == len(machines["harbor"]) == 1
    assert machines["docker"][0].network is NetworkPolicy.DENY
    assert machines["docker"][0].cpus is None
    assert machines["harbor"][0].cpus == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
async def test_disabled_harbor_verification_keeps_tokens_but_masks_training(
    tmp_path, task_inputs, phase, projection_type
):
    config, request = task_inputs
    source = tmp_path / "private"
    (source / "tests").mkdir(parents=True)
    (source / "instruction.md").write_text("Complete the task.")
    (source / "task.toml").write_text(
        '[environment]\nworkdir = "/workspace"\nallow_internet = false\ndocker_image = "fixture@sha256:'
        + "a" * 64
        + '"\n'
        '[verifier]\nenvironment_mode = "separate"\nuser = "candidate"\n'
    )
    (source / "tests/test.sh").write_text("echo private-grade-only\n")
    native = HarborTaskDataset(
        [str(source)], Tokenizer(), 100, session=session_spec(), cache_dir=tmp_path / "native", num_workers=1
    )
    assert LoweredTaskSpec.model_validate_json(native[0][2]["lowered_task_spec"]).task.resources.verifier
    settings = HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"verifier_disable": True}}))
    prepared = HarborTaskDataset(
        [str(source)],
        Tokenizer(),
        100,
        session=session_spec(),
        cache_dir=tmp_path / "disabled",
        verifier_override=settings.verifier_override(),
        num_workers=1,
    )
    prompt, env, extras, uid = prepared[0]
    assert LoweredTaskSpec.model_validate_json(extras["lowered_task_spec"]).task.resources.verifier == ()
    request.update(prompts=[prompt], env_classes=[env], env_extras=[extras], trajectory_ids=[TrajectoryID(uid, 0)])
    request["batch_metadata"] = BatchMetadata(0, phase)
    projection = (
        WholeTrajectoryProjection if projection_type is WholeTaskProjection else StepWiseTrajectoryProjection
    )(config, Tokenizer())

    class ImageFactory:
        async def create(self, spec):
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )

    worker = TaskRolloutWorker(
        config,
        projection_type(projection),
        ConversationClient(["Done"]),
        {"harbor": ImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    try:
        batch = await worker.run(request)
    finally:
        await worker.shutdown()
    assert all(grade.status == VerificationStatus.SKIPPED for grade in batch["verification_results"])
    assert all(grade.score is None for grade in batch["verification_results"])
    assert batch["response_ids"] == [[3, 4]]
    assert batch["loss_masks"] == [[0, 0]]
    assert batch["exclude_from_baseline"] == [True]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])
    assert batch["rollout_metrics"]["generate/task_rollout/turns"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_stage", ["startup", "model", "verifier", "attempt", "cancel", "cancel_model"])
async def test_harbor_retries_close_failed_attempts_and_commit_only_the_selected_result(task_inputs, failure_stage):
    config, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "tags": ("harbor",),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                parameters_json=ShellVerifierSpec(
                    argv=("cat", "/workspace/reward"),
                    artifacts=(
                        VerifierArtifact(
                            source="/workspace/reward",
                            target="/workspace/reward",
                            kind=ArtifactKind.FILE,
                            missing=MissingArtifactPolicy.SKIP,
                        ),
                    ),
                ).model_dump_json(),
            ),
        }
    )
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        task, "shellbox", backend="shellsim", verifier_backend="verifier"
    ).model_dump_json()
    attempts = 0
    machines = []
    waits = []
    backoff = asyncio.Event()

    class Factory:
        async def create(self, spec):
            nonlocal attempts
            attempts += 1
            if failure_stage == "startup" and attempts < 3:
                raise TimeoutError("Sandbox startup timed out")
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            machines.append(machine)
            if failure_stage not in {"verifier", "cancel"} or attempts == 3:
                await machine.run(Command(("sh", "-c", "echo 1 > /workspace/reward")))
            return machine

    class Client:
        async def generate(self, request):
            continuation = request["chat_continuations"][0]
            if failure_stage == "cancel_model":
                backoff.set()
                await asyncio.Future()
            if failure_stage == "attempt" and attempts < 3:
                await asyncio.Future()
            if failure_stage == "model" and continuation is not None and attempts < 3:
                raise ModelServerError("unavailable", "request-1", 503)
            message = {"role": "assistant", "content": "Done"}
            if failure_stage == "model" and continuation is None:
                message = {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "write",
                            "type": "function",
                            "function": {
                                "name": "shell",
                                "arguments": json.dumps(
                                    {
                                        "command": "test ! -f /workspace/dirty && echo retry > /workspace/dirty",
                                    }
                                ),
                            },
                        }
                    ],
                }
            offset = 0 if continuation is None else 2
            return {
                "assistant_messages": [message],
                "prompt_ids": [[1, 2] if continuation is None else continuation["served_prefix_token_ids"] + [90, 91]],
                "response_ids": [[10 * attempts + 3 + offset, 10 * attempts + 4 + offset]],
                "response_logprobs": [[-0.1, -0.2]],
                "stop_reasons": ["stop"],
                "token_provenance": TokenProvenance.ENGINE,
            }

    class VerifierFactory:
        async def create(self, spec):
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            machines.append(machine)
            return machine

    async def wait(delay):
        waits.append(delay)
        for machine in machines:
            with pytest.raises(RuntimeError, match="closed"):
                await machine.run(Command(("true",)))
        backoff.set()
        if failure_stage == "cancel":
            await asyncio.Future()

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "max_retries": 2,
                    "min_wait_sec": 1,
                    "wait_multiplier": 2,
                    "max_wait_sec": 1.5,
                    "include_exceptions": [
                        "EnvironmentStartTimeoutError",
                        "ModelServerError",
                        "VerifierRuntimeError",
                        "TrialTimeoutError",
                    ],
                    "trial_attempt_timeout_sec": 1 if failure_stage == "attempt" else None,
                    "enable_error_classification": True,
                    "default_error_treatment": "mask",
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {"shellsim": Factory(), "verifier": VerifierFactory()},
        harbor=settings,
        retry_wait=wait,
        shutdown_timeout=30,
    )
    writer = Writer()
    operation = asyncio.create_task(
        worker.run_task(
            RolloutTask(RolloutLease("lease", 0, 1), {"uid": task.id}, request),
            writer,
        )
    )
    if failure_stage in {"cancel", "cancel_model"}:
        await asyncio.wait_for(backoff.wait(), timeout=5)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert writer.groups == []
        assert attempts == 1
        for machine in machines:
            with pytest.raises(RuntimeError, match="closed"):
                await machine.run(Command(("true",)))
        await worker.shutdown()
        return
    await operation
    assert waits == [1, 1.5]
    assert attempts == 3
    assert len(writer.groups) == 1
    batch = writer.groups[0][1].trajectory_batch
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["rollout_metrics"]["rollout_retries"] == 2
    if failure_stage == "model":
        assert batch["response_ids"] == [[33, 34, 90, 91, 35, 36]]
        assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
    else:
        assert batch["response_ids"] == [[33, 34]]
        assert batch["loss_masks"] == [[1, 1]]
    for machine in machines:
        with pytest.raises(RuntimeError, match="closed"):
            await machine.run(Command(("true",)))


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "empty",
        "invalid",
        "retry_missing",
        "exhausted",
        "passthrough",
        "grade_timeout_keep",
        "grade_timeout_discard",
    ],
)
async def test_harbor_retry_policy_preserves_terminal_grades(task_inputs, phase, failure):
    config, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    failing_command = {
        "missing": "true",
        "retry_missing": "true",
        "empty": "printf '' > /reward.txt",
        "invalid": "echo invalid > /reward.txt",
        "exhausted": "exit 7",
        "passthrough": "echo 1 > /reward.txt",
        "grade_timeout_keep": "echo 1 > /reward.txt",
        "grade_timeout_discard": "echo 1 > /reward.txt",
    }[failure]
    script = (
        failing_command
        if failure == "exhausted"
        else f'if [ "$(cat /workspace/attempt)" -eq 1 ]; then {failing_command}; else echo 1 > /reward.txt; fi'
    )
    verifier = ShellVerifierSpec(
        argv=("sh", "-c", script),
        reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
        artifacts=(VerifierArtifact(source="/workspace/attempt", target="/workspace/attempt", kind=ArtifactKind.FILE),),
    )
    # A command failure without a reward file is a missing-reward failure.
    # Use stdout grading to exercise the retry limit for verifier execution errors.
    if failure == "exhausted":
        verifier = ShellVerifierSpec(argv=("false",), artifacts=verifier.artifacts)
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "tags": ("harbor",),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                parameters_json=verifier.model_dump_json(),
            ),
        }
    )
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        task,
        "shellbox",
        backend="shellsim",
        verifier_backend="verifier",
    ).model_dump_json()
    request["batch_metadata"] = BatchMetadata(0, phase)
    attempts = 0
    verifier_starts = 0
    waits = []

    class Factory:
        async def create(self, spec):
            nonlocal attempts
            attempts += 1
            machine = await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )
            await machine.run(Command(("sh", "-c", f"echo {attempts} > /workspace/attempt")))
            return machine

    class VerifierFactory:
        async def create(self, spec):
            nonlocal verifier_starts
            verifier_starts += 1
            if failure.startswith("grade_timeout"):
                raise TimeoutError("Verifier machine startup timed out")
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )

    class Client(InferenceClient):
        async def generate(self, request):
            if failure == "passthrough" and request["chat_continuations"][0] is not None:
                raise TimeoutError("Agent deadline exceeded")
            output = await super().generate(request)
            if failure == "passthrough":
                output["assistant_messages"] = [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "work",
                                "type": "function",
                                "function": {
                                    "name": "shell",
                                    "arguments": json.dumps({"command": "echo work"}),
                                },
                            }
                        ],
                    }
                ]
            return output

    async def wait(delay):
        waits.append(delay)

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "max_retries": 2,
                    "enable_error_classification": True,
                    "default_error_treatment": "mask",
                    "passthrough_exceptions": ["AgentTimeoutError"],
                    "preserve_logprobs_on_timeout": failure != "grade_timeout_discard",
                    **(
                        {"exclude_exceptions": [], "include_exceptions": ["RewardFileNotFoundError"]}
                        if failure == "retry_missing"
                        else {}
                    ),
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {"shellsim": Factory(), "verifier": VerifierFactory()},
        harbor=settings,
        retry_wait=wait,
        shutdown_timeout=30,
    )
    batch = await worker.run(request)
    retries = {"exhausted": 2, "retry_missing": 1}.get(failure, 0)
    assert attempts == retries + 1
    assert verifier_starts == retries + 1
    assert len(waits) == retries
    assert batch["rollout_metrics"]["rollout_retries"] == retries
    if failure in {"passthrough", "retry_missing"}:
        assert batch["unshaped_rewards"] == [1.0]
        assert batch["rewards"] == ([1.0] if failure == "passthrough" else [[0.0, 1.0]])
        assert batch["loss_masks"] == [[1, 1]]
        assert batch["exclude_from_baseline"] == [False]
    else:
        assert batch["unshaped_rewards"] == [0.0]
        assert batch["verification_results"][0].score is None
        assert batch["loss_masks"] == [[0, 0]]
        assert batch["exclude_from_baseline"] == [True]
    assert batch["response_ids"] == [[3, 4]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2]])


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_reason,expected", [("stop", 1.3), ("length", 0.8)])
async def test_harbor_completion_reward_uses_the_engine_stop_reason(task_inputs, stop_reason, expected):
    config, request = task_inputs
    task = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    verifier = ShellVerifierSpec(
        argv=("sh", "-c", "echo '===== 1 passed in 0.1s ====='; echo 1 > /reward.txt"),
        reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
    )
    task = task.model_copy(
        update={
            "answer_type": AnswerType.STATE,
            "tags": ("harbor",),
            "verifier": VerifierSpec(
                kind="shell",
                environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                parameters_json=verifier.model_dump_json(),
            ),
        }
    )
    request["env_extras"][0]["lowered_task_spec"] = lowered_task(
        task, "shellbox", backend="shellsim", verifier_backend="shellsim"
    ).model_dump_json()

    class Client(InferenceClient):
        async def generate(self, request):
            result = await super().generate(request)
            result["stop_reasons"] = [stop_reason]
            return result

    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "enable_reward_shaping": True,
                    "reward_shaper": "composite_loop",
                    "reward_parser": "pytest",
                    "loop_shaping": {"terminate": {"enabled": True}},
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        Client(),
        {"shellsim": FixtureImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    batch = await worker.run(request)
    assert batch["unshaped_rewards"] == [1.0]
    assert batch["rewards"][0] == pytest.approx([0.0, expected])
    assert batch["response_ids"] == [[3, 4]]
    assert batch["loss_masks"] == [[1, 1]]


@pytest.mark.asyncio
@pytest.mark.parametrize("shaper", ["threshold", "identity_aware_pass_ratio"])
async def test_strict_harbor_parser_masks_only_the_affected_response(task_inputs, shaper):
    config, request = task_inputs
    base = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    tasks = []
    for output in ("unrecognized output", "tests/test_task.py::test_answer PASSED"):
        verifier = ShellVerifierSpec(
            argv=("sh", "-c", f"echo '{output}'; echo 1 > /reward.txt"),
            reward=FileReward(files=(RewardFile(path="/reward.txt", format=RewardFileFormat.NUMBER),)),
        )
        tasks.append(
            base.model_copy(
                update={
                    "answer_type": AnswerType.STATE,
                    "tags": ("harbor",),
                    "verifier": VerifierSpec(
                        kind="shell",
                        environment_requirements=EnvironmentRequirements(docker_image="fixture@sha256:" + "0" * 64),
                        parameters_json=verifier.model_dump_json(),
                    ),
                }
            )
        )
    request.update(
        prompts=request["prompts"] * 2,
        env_classes=["taskcompendium"] * 2,
        env_extras=[
            {
                "lowered_task_spec": lowered_task(
                    task, "shellbox", backend="shellsim", verifier_backend="shellsim"
                ).model_dump_json()
            }
            for task in tasks
        ],
        trajectory_ids=[TrajectoryID(base.id, index) for index in range(2)],
    )
    settings = HarborTaskSettings.from_config(
        OmegaConf.create(
            {
                "harbor": {
                    "enable_reward_shaping": True,
                    "reward_shaper": shaper,
                    "reward_parser": "pytest",
                    "reward_shaping_fallback": False,
                    "enable_error_classification": True,
                }
            }
        )
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        InferenceClient(),
        {"shellsim": FixtureImageFactory()},
        harbor=settings,
        shutdown_timeout=30,
    )
    batch = await worker.run(request)
    assert batch["unshaped_rewards"] == [1.0, 1.0]
    assert [verdict.score for verdict in batch["verification_results"]] == [1.0, 1.0]
    assert batch["response_ids"] == [[3, 4], [3, 4]]
    assert batch["loss_masks"] == [[0, 0], [1, 1]]
    assert batch["exclude_from_baseline"] == [True, False]
    assert batch["exception_types"] == ["VerifierOutputParseError", None]


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["train", "eval"])
async def test_harbor_concurrency_does_not_queue_gym_tasks(task_inputs, phase):
    config, request = task_inputs
    base = LoweredTaskSpec.model_validate_json(request["env_extras"][0]["lowered_task_spec"]).task
    tasks = [base.model_copy(update={"id": "harbor", "tags": ("harbor",)})] * 2 + [base]
    admitted = []
    active_harbor = 0
    maximum_harbor = 0
    ready = asyncio.Event()
    release = asyncio.Event()

    class WaitingClient(InferenceClient):
        async def generate(self, request):
            nonlocal active_harbor, maximum_harbor
            is_harbor = request["prompts"][0][0]["content"].startswith("Terminal task")
            admitted.append(is_harbor)
            if is_harbor:
                active_harbor += 1
                maximum_harbor = max(maximum_harbor, active_harbor)
            if len(admitted) == (2 if phase == "train" else 3):
                ready.set()
            await release.wait()
            if is_harbor:
                active_harbor -= 1
            return await super().generate(request)

    tasks[:2] = [
        task.model_copy(
            update={"context": ConversationInput(events=(TextMessage(role="user", content="Terminal task"),))}
        )
        for task in tasks[:2]
    ]
    request.update(
        prompts=request["prompts"] * 3,
        env_classes=["taskcompendium"] * 3,
        env_extras=[{"lowered_task_spec": lowered_task(task, "shellbox").model_dump_json()} for task in tasks],
        trajectory_ids=[TrajectoryID(str(index), 0) for index in range(3)],
        batch_metadata=BatchMetadata(0, phase),
    )
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        WaitingClient(),
        {},
        harbor=HarborTaskSettings.from_config(OmegaConf.create({"harbor": {"n_concurrent_trials": 2}})),
        concurrent_tasks=2 if phase == "train" else 3,
        concurrent_harbor_tasks=1,
        shutdown_timeout=30,
    )
    operation = asyncio.create_task(worker.run(request))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert admitted.count(False) == 1
        assert maximum_harbor == (1 if phase == "train" else 2)
    finally:
        release.set()
        try:
            batch = await operation
        finally:
            await worker.shutdown()
    assert batch["unshaped_rewards"] == [1.0, 1.0, 1.0]


@pytest.mark.asyncio
async def test_harbor_source_materialization_runs_without_the_original_directory(tmp_path, task_inputs):
    source = tmp_path / "source"
    (source / "environment").mkdir(parents=True)
    (source / "tests").mkdir()
    (source / "instruction.md").write_text("Write 12 to /logs/artifacts/answer.")
    (source / "task.toml").write_text(
        '[environment]\nworkdir = "/workspace"\nallow_internet = false\ndocker_image = "fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n'
        '[verifier]\nenvironment_mode = "separate"\n'
    )
    (source / "tests/test.sh").write_text(
        '#!/bin/sh\nif [ "$(cat /logs/artifacts/answer)" = "12" ]; then\n'
        "echo 0.75 > /logs/verifier/reward.txt\nelse\necho 0 > /logs/verifier/reward.txt\nfi\n"
    )
    prepared = HarborTaskDataset(
        [str(source)], Tokenizer(), 100, session=session_spec(), cache_dir=tmp_path / "tasks", num_workers=1
    )
    assert prepared.uid(0) == "source"
    shutil.rmtree(source)
    restored = TaskDataset([str(prepared.task_path)], Tokenizer(), 100, num_workers=1)
    prompt, env, extras, uid = restored[0]

    class ImageFactory:
        async def create(self, spec):
            return await ShellSimMachineFactory().create(
                replace(spec, source=ShellSimBuiltins(), workdir=spec.workdir or "/workspace")
            )

    client = ConversationClient(
        ["", "Done"],
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "answer",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "echo 12 > /logs/artifacts/answer"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Done"},
        ],
    )
    config, request = task_inputs
    request.update(prompts=[prompt], env_classes=[env], env_extras=[extras], trajectory_ids=[TrajectoryID(uid, 0)])
    worker = TaskRolloutWorker(
        config,
        WholeTaskProjection(WholeTrajectoryProjection(config, Tokenizer())),
        client,
        {"docker": ImageFactory()},
        shutdown_timeout=30,
    )
    writer = Writer()
    await worker.run_task(RolloutTask(RolloutLease("lease", 0, 1), {"uid": uid}, request), writer)
    batch = writer.groups[0][1].trajectory_batch
    assert batch["unshaped_rewards"] == [0.75]
    assert batch["rollout_metrics"]["generate/task_rollout/tool_tasks"] == 1.0
    assert batch["response_ids"] == [[3, 4, 90, 91, 5, 6]]
    assert batch["loss_masks"] == [[1, 1, 0, 0, 1, 1]]
    np.testing.assert_allclose(batch["rollout_logprobs"], [[-0.1, -0.2, 0, 0, -0.1, -0.2]])
    assert all("reward.txt" not in json.dumps(request["prompts"]) for request in client.requests)
