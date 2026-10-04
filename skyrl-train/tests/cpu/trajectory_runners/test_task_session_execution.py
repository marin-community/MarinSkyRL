import pickle
from functools import partial

import hydra
import pytest
from skyrl_gym.answer_tasks import grade_gsm8k
from skyrl_gym.task_sessions import AnswerTaskSession
from skyrl_gym.verification import VerificationStatus
from rolloutengine.contracts import ModelTurn
from taskcompendium.grading import Outcome
from transformers import AutoTokenizer
from taskcompendium.importers.skyrl import source_task
from taskcompendium.models import Source

from skyrl_train.entrypoints.main_base import config_dir
from skyrl_train.rollouts.workers import WorkerShard
from skyrl_train.rollouts.task_worker import TaskRolloutWorkerSpec
from skyrl_train.trajectory_runners.types import TrajectoryID
from tests.cpu.tiny_training.cpu_backend import CPUInferenceEngine
from tests.cpu.tiny_training.tiny_model import build_tiny_policy
from examples.multiply.task_session import MultiplyTaskSession
from examples.llm_as_a_judge.task_grading import grade_judged_answer


@pytest.mark.asyncio
@pytest.mark.parametrize("session,max_turns,rewards", [("custom_math", 1, [0.0]), ("gsm8k_multi_turn", 2, [0.1, 0.1])])
async def test_pickled_worker_runs_real_cpu_inference_with_direct_sessions(tmp_path, session, max_turns, rewards):
    model_path = build_tiny_policy(tmp_path / "model")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    with hydra.initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = hydra.compose(config_name="ppo_base_config")
    cfg.trainer.policy.model.path = str(model_path)
    cfg.trainer.step_wise_training = True
    cfg.generator.max_turns = max_turns
    cfg.generator.max_input_length = 256
    cfg.generator.sampling_params.temperature = 0
    cfg.generator.sampling_params.max_generate_length = 16
    cfg.generator.sampling_params.logprobs = 0
    task = source_task(
        [{"role": "user", "content": "What is two? End with #### 2."}],
        session,
        {"reward_spec": {"ground_truth": "2"}},
        {},
        Source(dataset="tiny", revision="1", row="0", importer_revision="1"),
    )
    spec = TaskRolloutWorkerSpec.from_config(
        cfg,
        [CPUInferenceEngine(str(model_path), seed=0)],
        sessions={"custom_math": partial(AnswerTaskSession, grader=grade_gsm8k)},
    )
    worker = pickle.loads(pickle.dumps(spec)).build(tokenizer, WorkerShard(index=0, count=1))
    try:
        batch = await worker.run(
            {
                "prompts": [[{"role": "user", "content": "What is two? End with #### 2."}]],
                "env_classes": [session],
                "env_extras": [{"task_spec": task.model_dump_json()}],
                "trajectory_ids": [TrajectoryID(task.id, 0)],
                "batch_metadata": None,
                "sampling_params": None,
            }
        )
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
        "multiply",
        {"reward_spec": {"ground_truth": "42"}, "max_turns": 2},
        {},
        Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
    )
    session = MultiplyTaskSession(task, None)
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
        assert (await session.grade(())).reward == 0.5
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
