import json
import pickle
from functools import partial

import hydra
import pytest
from omegaconf import OmegaConf
from skyrl_gym.answer_tasks import grade_gsm8k
from skyrl_gym.task_sessions import AnswerTaskSession
from skyrl_gym.verification import VerificationStatus
from rolloutengine.contracts import ModelTurn
from taskcompendium.grading_result import Outcome
from transformers import AutoTokenizer
from skyrl_gym.source_task import source_task
from taskcompendium.models import NoGrader, Source
from shellbox.backends.daytona.machine import DaytonaMachineFactory
from tests.cpu.task_specs import lowered_task

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.dataset.tasks import source_row_task
from skyrl_train.rollouts.buffer import RolloutGroup, RolloutLease, RolloutTask
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.trajectory_runners.types import TrajectoryID
from skyrl_train.utils.utils import validate_cfg
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import build_tiny_policy
from examples.multiply.task_session import MultiplyTaskSession
from examples.llm_as_a_judge.task_grading import grade_judged_answer


class RecordingWriter:
    def __init__(self):
        self.groups: list[RolloutGroup] = []

    async def write_rollout(self, lease: RolloutLease, group: RolloutGroup) -> None:
        self.groups.append(group)


@pytest.mark.asyncio
@pytest.mark.parametrize("harbor_backend", ["docker", "daytona"])
@pytest.mark.parametrize("missing_executable", ["docker", "skopeo", None])
@pytest.mark.parametrize("session,max_turns,rewards", [("custom_math", 1, [0.0]), ("gsm8k_multi_turn", 2, [0.0, 0.0])])
async def test_cpu_worker_runs_machine_free_rows_and_checks_native_tools(
    tmp_path, monkeypatch, session, max_turns, rewards, missing_executable, harbor_backend
):
    model_path = build_tiny_policy(tmp_path / "model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.off_policy_correction = "tis"
    cfg.trainer.step_wise_training = True
    cfg.generator.max_turns = max_turns
    cfg.generator.max_input_length = 256
    cfg.trainer.max_prompt_length = 256
    cfg.generator.sampling_params.temperature = 0.01
    cfg.generator.sampling_params.max_generate_length = 16
    cfg.generator.sampling_params.logprobs = None
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in {"docker", "skopeo"} - {missing_executable}:
        executable = tools / name
        executable.write_text("#!/bin/sh\nexit 1\n")
        executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tools))
    cfg.trajectory_runner.skopeo = str(tools / "skopeo")
    cfg.trajectory_runner.image_cache = str(tmp_path / "image-cache")
    validate_cfg(cfg)

    async def unavailable_daytona(_factory, _spec):
        raise ConnectionError("The remote provider is unavailable")

    monkeypatch.setattr(DaytonaMachineFactory, "create", unavailable_daytona)
    task = source_task(
        [{"role": "user", "content": "What is two? End with #### 2."}],
        {"reward_spec": {"ground_truth": "2"}},
        {},
        Source(dataset="tiny", revision="1", row="0", importer_revision="1"),
    )
    spec = TaskRolloutWorkerSpec.from_config(
        cfg,
        [CPUInferenceEngine(str(model_path), seed=0)],
        sessions={"custom_math": partial(AnswerTaskSession, grader=grade_gsm8k)},
        harbor_config=OmegaConf.create(
            {"harbor": {"environment_type": harbor_backend, "n_concurrent_trials": 1, "max_retries": 0}}
        ),
    )
    worker = pickle.loads(pickle.dumps(spec)).build(tokenizer, WorkerShard(index=0, count=1))
    try:
        request = {
            "prompts": [[{"role": "user", "content": "What is two? End with #### 2."}]],
            "env_classes": [session],
            "env_extras": [{"lowered_task_spec": lowered_task(task, session, max_turns=max_turns).model_dump_json()}],
            "trajectory_ids": [TrajectoryID(task.id, 0)],
            "batch_metadata": None,
            "sampling_params": None,
        }
        batch = await worker.run(request)
        native_row = {
            "prompt": [{"role": "user", "content": "Write a Python program that prints 2."}],
            "env_class": "lcb",
            "reward_model": {"ground_truth": json.dumps([{"input": "", "output": "2", "testtype": "stdin"}])},
        }
        native = source_row_task(
            native_row,
            1,
            source_name="tiny",
            environment_configs=OmegaConf.to_container(cfg.environment.task_sessions, resolve=True),
        )
        harbor_task = native.model_copy(
            update={
                "task": native.task.model_copy(
                    update={
                        "tags": ("harbor",),
                        "grader": NoGrader(reason="fixture"),
                    }
                ),
                "session": native.session.model_copy(update={"task_session": "shellbox"}),
            }
        )
        for backend, candidate in (("docker", native), ("harbor", harbor_task)):
            mixed = {
                **request,
                "prompts": [*request["prompts"], native_row["prompt"]],
                "env_classes": [session, candidate.session.task_session],
                "env_extras": [*request["env_extras"], {"lowered_task_spec": candidate.model_dump_json()}],
                "trajectory_ids": [TrajectoryID(task.id, 0), TrajectoryID(candidate.task.id, 0)],
            }
            writer = RecordingWriter()
            operation = RolloutTask(RolloutLease("mixed", 0, 0), {"uid": "mixed"}, mixed)
            if missing_executable is None or (backend == "harbor" and harbor_backend == "daytona"):
                await worker.run_task(operation, writer)
                assert len(writer.groups) == 1
                mixed_batch = writer.groups[0].trajectory_batch
                assert mixed_batch["exception_types"][-1] == "TaskMachineError"
                assert mixed_batch["loss_masks"][-1] == [0]
                assert mixed_batch["exclude_from_baseline"][-1] is True
            else:
                with pytest.raises(ValueError, match=f"Unknown Shellbox factory: {backend!r}"):
                    await worker.run_task(operation, writer)
                assert writer.groups == []
    finally:
        await worker.shutdown()
    expected_response = tokenizer.encode("#### 1<|im_end|>", add_special_tokens=False)
    assert batch["response_ids"] == [expected_response] * max_turns
    if max_turns == 2:
        first_prefix = batch["prompt_token_ids"][0] + expected_response
        assert batch["prompt_token_ids"][1][: len(first_prefix)] == first_prefix
    assert batch["loss_masks"] == [[1] * len(expected_response)] * max_turns
    assert all(len(scores) == len(expected_response) for scores in batch["rollout_logprobs"])
    assert batch["unshaped_rewards"] == rewards
    assert all(result.status == VerificationStatus.VERIFIED for result in batch["verification_results"])


@pytest.mark.asyncio
async def test_multiplication_session_returns_feedback_then_rewards_the_final_answer():
    task = source_task(
        [{"role": "user", "content": "What is six times seven?"}],
        {"reward_spec": {"ground_truth": "42"}, "max_turns": 2},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    session = MultiplyTaskSession(lowered_task(task, "multiply", max_turns=2), None)
    await session.prepare()
    try:
        results = []
        for answer in (41, 42):
            text = rf"\boxed{{{answer}}}"
            results.append(
                await session.advance(
                    ModelTurn({"role": "assistant", "content": text}, (), (1,), None, "stop", text=text)
                )
            )
        assert not results[0].done and results[0].reward == 0.0
        assert results[0].observations[0]["role"] == "user"
        assert results[1].done and results[1].reward == 1.0
        assert results[1].observations == ()
        assert (await session.grade(())).reward == 1.0
        assert (await session.grade(())).passed is True
    finally:
        await session.close()


@pytest.mark.parametrize(
    "reply,reward,status",
    [
        ("### Final Score: 1", 1.0, Outcome.GRADED),
        ("0", 0.0, Outcome.GRADED),
        ("No verdict available", None, Outcome.INFRA_ERROR),
    ],
)
def test_judge_example_keeps_missing_verdicts_ungraded(monkeypatch, reply, reward, status):
    class Reply:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"finish_reason": "stop", "message": {"content": reply}}]}

    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.setattr("requests.post", lambda *args, **kwargs: Reply())
    turn = ModelTurn({"role": "assistant", "content": "#### 12"}, (), (1,), None, "stop", text="#### 12")
    result = grade_judged_answer(
        turn,
        {"base_url": "https://judge.example/v1", "model": "fixture"},
        {"reward_spec": {"ground_truth": "12"}},
    )
    assert result.grade.status is status
    assert result.grade.reward == reward
