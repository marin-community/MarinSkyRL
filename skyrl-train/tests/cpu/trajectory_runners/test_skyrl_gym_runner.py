"""
uv run --group dev --extra cpu --isolated pytest tests/cpu/trajectory_runners/test_skyrl_gym_runner.py
"""

import torch
from skyrl_train.config.objective_spec import load_correction
from skyrl_train.objective.correction import compute_correction
from concurrent.futures import Executor, Future
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import skyrl_gym
from loguru import logger
from omegaconf import DictConfig
from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput
from skyrl_gym.verification import RewardResult, RolloutEvidence, TrainingDisposition, VerificationResult

from marinskyrl.distillation import TeacherEvidenceKind
from skyrl_train.config.utils import get_default_config
from skyrl_train.distillation_adapters import build_teacher_scoring_work
from skyrl_train.rollout_observability import observe_rollout_call
from skyrl_train.trajectory_runners.base import TrajectoryID, TrajectoryRequestBatch
from skyrl_train.trajectory_runners.model_clients import ModelServerError
from skyrl_train.trajectory_runners.skyrl_gym import ExactChatTransportError, SkyRLGymTrajectoryRunner
from skyrl_train.trajectory_runners.trajectory_processing import (
    normalize_token_ids,
    validate_trajectory_batch as assert_valid_trajectory_batch,
)
from skyrl_train.trajectory_runners.types import AgentLoopOutput, BatchMetadata, TokenProvenance
from skyrl_train.utils.utils import validate_cfg

QWEN2_5 = "Qwen/Qwen2.5-0.5B-Instruct"
# `<|im_end|>`, the Qwen2.5 instruct EOS. Parametrized expectations name it before the tokenizer fixture exists.
EOS = 151645
ENV_CLASS = "gsm8k"
SAMPLED_IDS = [10, 12, EOS]


@pytest.fixture
def tokenizer(load_tokenizer):
    tokenizer = load_tokenizer(QWEN2_5)
    assert tokenizer.eos_token_id == EOS
    return tokenizer


@pytest.fixture
def mock_llm():
    """An inference engine client that samples `SAMPLED_IDS` for every prompt."""
    mock = MagicMock()

    def mock_generate(input_batch):
        num_prompts = len(input_batch["prompts"]) if "prompts" in input_batch else len(input_batch["prompt_token_ids"])
        return {
            "responses": ["mocked output"] * num_prompts,
            "stop_reasons": ["stop"] * num_prompts,
            "response_logprobs": [[0.1] * len(SAMPLED_IDS)] * num_prompts,
            "response_ids": [SAMPLED_IDS.copy()] * num_prompts,
        }

    mock.generate = AsyncMock(side_effect=mock_generate)
    return mock


def engine_returning(*outputs: dict) -> MagicMock:
    engine = MagicMock()
    engine.generate = AsyncMock(side_effect=list(outputs))
    return engine


@pytest.fixture
def mock_env():
    mock_env_instance = MagicMock()
    mock_env_instance.step.side_effect = lambda x: BaseTextEnvStepOutput(
        observations=[{"role": "user", "content": "next"}], reward=1.0, done=True, metadata={}
    )
    mock_env_instance.close.return_value = None
    return mock_env_instance


class ScriptedEnv(BaseTextEnv):
    """Environment that replays step outputs, repeating the last one once the script runs out."""

    def __init__(self, *steps: BaseTextEnvStepOutput):
        super().__init__()
        self.steps = steps
        self.num_steps = 0

    def init(self, prompt):
        return prompt, {}

    def step(self, action):
        self.num_steps += 1
        return self.steps[min(self.num_steps, len(self.steps)) - 1]


@pytest.fixture
def use_env(monkeypatch):
    def install(env):
        monkeypatch.setattr(skyrl_gym, "make", lambda *_args, **_kwargs: env)
        return env

    return install


@pytest.fixture
def generator_cfg():
    cfg = get_default_config().generator
    cfg.sampling_params.max_generate_length = 5
    cfg.sampling_params.logprobs = None
    cfg.apply_overlong_filtering = False
    cfg.max_input_length = 512
    cfg.max_turns = 1
    cfg.chat_template_kwargs = {}
    cfg.chat_template = {"source": "name", "name_or_path": None}
    return cfg


@pytest.fixture
def skyrl_gym_cfg():
    return DictConfig({"max_env_workers": 0})


def single_prompt_request(content: str = "question") -> TrajectoryRequestBatch:
    return {"prompts": [[{"role": "user", "content": content}]], "env_extras": [{}], "env_classes": [ENV_CLASS]}


def _successful_trajectory_output() -> AgentLoopOutput:
    return AgentLoopOutput(
        evidence=RolloutEvidence(
            response="ok",
            stop_reason="stop",
            generated_token_count=1,
            prompt_token_ids=(11,),
            response_token_ids=(12,),
            behavior_logprobs=np.asarray([-0.5], dtype=np.float32),
            student_topk_indices=np.asarray([[12, 13]], dtype=np.int32),
            behavior_topk_logprobs=np.asarray([[-0.5, -1.5]], dtype=np.float32),
        ),
        verification=VerificationResult.verified(1.0, passed=True),
        reward=RewardResult(unshaped_reward=1.0, optimization_reward=1.0, token_rewards=(1.0,)),
        disposition=TrainingDisposition.train(),
        loss_mask=[1],
        env_metrics={},
    )


def _masking_agent_loop(error: Exception):
    async def agent_loop(prompt, *_args, **_kwargs):
        if prompt[0]["content"] == "fail":
            raise error
        return _successful_trajectory_output()

    return agent_loop


def _two_row_request(training_phase: str) -> TrajectoryRequestBatch:
    return TrajectoryRequestBatch(
        prompts=[
            [{"role": "user", "content": "ok"}],
            [{"role": "user", "content": "fail"}],
        ],
        env_classes=[ENV_CLASS, ENV_CLASS],
        env_extras=[{}, {}],
        sampling_params=None,
        trajectory_ids=[TrajectoryID("ok", 0), TrajectoryID("fail", 0)],
        batch_metadata=BatchMetadata(global_step=1, training_phase=training_phase),
    )


@pytest.mark.asyncio
async def test_whole_trajectory_collector_masks_one_agent_loop_failure(generator_cfg, skyrl_gym_cfg, tokenizer):
    generator_cfg.sampling_params.logprobs = 1
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    runner.agent_loop = _masking_agent_loop(TimeoutError("judge request timed out"))

    batch = await runner._run(_two_row_request("train"), disable_tqdm=True)

    assert batch["response_ids"] == [[12], [0]]
    assert batch["rewards"] == [[1.0], [0.0]]
    assert batch["loss_masks"] == [[1], [0]]
    np.testing.assert_allclose(batch["rollout_logprobs"][0], [-0.5])
    np.testing.assert_allclose(batch["rollout_logprobs"][1], [0.0])
    np.testing.assert_array_equal(batch["student_topk_indices"][0], [[12, 13]])
    np.testing.assert_array_equal(batch["student_topk_indices"][1], [[-1, -1]])
    np.testing.assert_array_equal(batch["behavior_topk_logprobs"][1], [[0.0, 0.0]])
    assert batch["exclude_from_baseline"] == [False, False]
    assert batch["exception_types"] == [None, "AgentTimeoutError"]
    assert batch["error_treatments"] == [None, "zero"]


@pytest.mark.asyncio
@patch("skyrl_gym.make")
@pytest.mark.parametrize(
    ("failure_phase", "treatment"),
    [("generate", "zero"), ("step", "zero"), ("generate", "passthrough"), ("generate", "mask")],
)
async def test_gym_terminal_error_retains_only_completed_turn(
    mock_make, generator_cfg, skyrl_gym_cfg, tokenizer, mock_env, failure_phase, treatment
):
    generator_cfg.use_conversation_multi_turn = False
    generator_cfg.sampling_params.logprobs = 1
    if treatment == "passthrough":
        generator_cfg.error_handling.passthrough_exceptions = ["ContextLengthExceededError"]
    elif treatment == "mask":
        generator_cfg.error_handling.mask_exceptions = ["ContextLengthExceededError"]
    mock_env.init.return_value = ([{"role": "user", "content": "question"}], {})
    mock_env.step.side_effect = [
        BaseTextEnvStepOutput(observations=[{"role": "user", "content": "next"}], reward=1.0, done=False, metadata={}),
        TimeoutError("step timed out") if failure_phase == "step" else None,
    ]
    mock_make.return_value = mock_env
    model_client = AsyncMock()
    successful_turn = {
        "responses": ["answer"],
        "response_ids": [[10, 12]],
        "stop_reasons": ["stop"],
        "response_logprobs": [[-0.1, -0.2]],
        "routed_experts": [np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8)],
        "token_provenance": "engine",
    }
    model_client.generate.side_effect = [
        successful_turn,
        ModelServerError("context_overflow", "request-123", 400) if failure_phase == "generate" else successful_turn,
    ]
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer, model_client=model_client)

    output = await runner.agent_loop([{"role": "user", "content": "question"}], "test", {}, 8, 512)

    assert output.evidence.response_token_ids[:2] == (10, 12)
    np.testing.assert_allclose(output.evidence.behavior_logprobs[:2], [-0.1, -0.2])
    np.testing.assert_array_equal(output.evidence.routed_experts[:2], [[[1, 2]], [[3, 4]]])
    assert output.evidence.generated_token_count == 2
    assert output.verification.score == 1.0
    if failure_phase == "generate":
        assert output.verification.diagnostics["request_id"] == "request-123"
    assert output.reward.unshaped_reward == 1.0
    assert output.reward.optimization_reward == (1.0 if treatment == "passthrough" else 0.0)
    assert output.disposition.exception_type == (
        "ContextLengthExceededError" if failure_phase == "generate" else "AgentTimeoutError"
    )
    assert output.error_treatment == treatment
    assert output.disposition.loss_eligible is (treatment != "mask")
    assert output.disposition.baseline_eligible is (treatment != "mask")


@pytest.mark.asyncio
async def test_gym_server_failure_is_masked_with_safe_diagnostics(generator_cfg, skyrl_gym_cfg, tokenizer):
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    runner.agent_loop = _masking_agent_loop(ModelServerError("constrained_decoding", "request-123", 500))

    batch = await runner._run(_two_row_request("train"), disable_tqdm=True)

    assert batch["loss_masks"][1] == [0]
    assert batch["server_errors"] == [
        None,
        {"category": "constrained_decoding", "request_id": "request-123", "status_code": 500},
    ]
    assert batch["verification_results"][1].diagnostics == {
        "exception_type": "ModelServerError",
        "error_category": "constrained_decoding",
        "request_id": "request-123",
        "status_code": 500,
    }


@pytest.mark.asyncio
async def test_whole_trajectory_collector_propagates_exact_chat_contract_failure(
    generator_cfg, skyrl_gym_cfg, tokenizer
):
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    runner.agent_loop = _masking_agent_loop(ExactChatTransportError("served prefix changed"))

    with pytest.raises(ExactChatTransportError, match="served prefix changed"):
        await runner._run(_two_row_request("train"), disable_tqdm=True)


@pytest.mark.asyncio
async def test_whole_trajectory_collector_adapts_masked_scalar_rewards_to_token_level_batch(
    generator_cfg, skyrl_gym_cfg, tokenizer
):
    """A masked agent-loop failure must not change a token-level batch's reward form.

    https://github.com/marin-community/MarinSkyRL/issues/680
    """
    # In retokenize mode the failed-row producer emits a scalar reward, while successful
    # chat-completions trajectories in the same batch carry token-level rewards.
    generator_cfg.use_conversation_multi_turn = True
    generator_cfg.chat_template = {"source": "name", "name_or_path": "qwen3_without_thinking"}
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    runner.agent_loop = _masking_agent_loop(ConnectionError("judge returned HTTP 503"))

    batch = await runner._run(_two_row_request("eval"), disable_tqdm=True)

    assert_valid_trajectory_batch(2, batch)
    assert batch["rewards"] == [[1.0], [0.0]]
    assert batch["loss_masks"] == [[1], [0]]
    assert batch["exclude_from_baseline"] == [False, True]
    assert batch["exception_types"] == [None, "ConnectionError"]
    assert batch["error_treatments"] == [None, "mask"]


@pytest.mark.asyncio
async def test_agent_loop_failure_closes_environment_before_masking(generator_cfg, skyrl_gym_cfg, tokenizer):
    env = MagicMock()
    env.init.side_effect = TimeoutError("environment initialization timed out")
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    request = TrajectoryRequestBatch(
        prompts=[[{"role": "user", "content": "fail"}]],
        env_classes=[ENV_CLASS],
        env_extras=[{}],
        sampling_params=None,
        trajectory_ids=[TrajectoryID("fail", 0)],
        batch_metadata=BatchMetadata(global_step=1, training_phase="train"),
    )

    with patch("skyrl_train.trajectory_runners.skyrl_gym.skyrl_gym.make", return_value=env):
        batch = await runner._run(request, disable_tqdm=True)

    env.close.assert_called_once_with()
    assert batch["response_ids"] == [[0]]
    assert batch["loss_masks"] == [[0]]
    assert batch["exception_types"] == ["AgentTimeoutError"]


def test_tis_config_does_not_select_a_generation_strategy():
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    cfg.trainer.algorithm.off_policy_correction = "tis"
    cfg.generator.sampling_params.logprobs = None

    validate_cfg(cfg)

    assert cfg.generator.sampling_params.logprobs == 0


@pytest.mark.asyncio
async def test_genrm_rewards_replace_provisional_rewards_by_prompt_cohort(generator_cfg, tokenizer):
    skyrl_gym_cfg = DictConfig(
        {
            "max_env_workers": 0,
            "nemotron_ultra": {
                "genrm": {
                    "num_rollouts_per_prompt": 2,
                    "group_answer_length_penalty_coeff": 0.0,
                    "group_reasoning_length_penalty_coeff": 0.0,
                    "reasoning_bonus": 0.0,
                    "answer_bonus": 0.0,
                }
            },
        }
    )
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)

    class _GenRMJudge:
        def generate_response(self, _messages, *, metadata, **_kwargs):
            first_is_better = metadata["response_1"] == "better"
            return (
                '{"score_1": 5, "score_2": 1, "ranking": 1}'
                if first_is_better
                else '{"score_1": 1, "score_2": 5, "ranking": 6}'
            )

    runner.genrm_judge = _GenRMJudge()

    def output(answer):
        return AgentLoopOutput(
            evidence=RolloutEvidence(
                messages=(
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "unparsed reasoning then " + answer},
                ),
                response=answer,
                response_token_ids=(10, 11),
            ),
            verification=VerificationResult.verified(3.0),
            reward=RewardResult(unshaped_reward=3.0, optimization_reward=3.0, token_rewards=(0.0, 3.0)),
            disposition=TrainingDisposition.train(),
            loss_mask=[1, 1],
            env_metrics={},
        )

    outputs = [output("better"), output("worse")]
    ultra = {
        "agent": "genrm_simple_agent",
        "record_json": '{"principle": "Prefer correct answers."}',
    }
    request = {
        "prompts": [[{"role": "user", "content": "q"}]] * 2,
        "env_classes": ["nemotron_ultra"] * 2,
        "env_extras": [{"extra_info": {"nemotron_ultra": ultra}}] * 2,
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID("prompt", 0), TrajectoryID("prompt", 1)],
        "batch_metadata": None,
    }

    await runner._apply_genrm_cohort_rewards(outputs, request)

    assert [item.reward.optimization_reward for item in outputs] == pytest.approx([5.0, 1.0])
    assert [item.reward.token_rewards for item in outputs] == [(0.0, 5.0), (0.0, 1.0)]
    assert outputs[0].evidence.messages[-1]["content"] == "unparsed reasoning then better"


@pytest.mark.asyncio
async def test_genrm_cohort_ranking_is_skipped_for_single_sample_evaluation(generator_cfg, tokenizer):
    skyrl_gym_cfg = DictConfig({"max_env_workers": 0, "nemotron_ultra": {"genrm": {"num_rollouts_per_prompt": 16}}})
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    runner.genrm_judge = MagicMock()
    output = AgentLoopOutput(
        evidence=RolloutEvidence(
            messages=({"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}),
            response="answer",
            response_token_ids=(10, 11),
        ),
        verification=VerificationResult.verified(3.0),
        reward=RewardResult(unshaped_reward=3.0, optimization_reward=3.0, token_rewards=(0.0, 3.0)),
        disposition=TrainingDisposition.train(),
        loss_mask=[1, 1],
        env_metrics={},
    )
    ultra = {
        "agent": "genrm_simple_agent",
        "record_json": '{"principle": "Prefer correct answers."}',
    }
    request = {
        "prompts": [[{"role": "user", "content": "q"}]],
        "env_classes": ["nemotron_ultra"],
        "env_extras": [{"extra_info": {"nemotron_ultra": ultra}}],
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID("prompt", 0)],
        "batch_metadata": BatchMetadata(global_step=0, training_phase="eval"),
    }

    await runner._apply_genrm_cohort_rewards([output], request)

    runner.genrm_judge.generate_response.assert_not_called()
    assert output.reward.optimization_reward == 0.0
    assert not output.disposition.loss_eligible
    assert output.verification.status.value == "unavailable"
    assert output.env_metrics["genrm/cohort_skipped_eval"] == 1.0


@pytest.mark.asyncio
async def test_genrm_cohort_ranking_is_skipped_when_grading_is_skipped(generator_cfg, tokenizer):
    skyrl_gym_cfg = DictConfig(
        {"max_env_workers": 0, "nemotron_ultra": {"grading": "skip", "genrm": {"num_rollouts_per_prompt": 1}}}
    )
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    runner.genrm_judge = MagicMock()
    verification = VerificationResult.skipped("grading is skipped")
    output = AgentLoopOutput(
        evidence=RolloutEvidence(
            messages=({"role": "user", "content": "q"}, {"role": "assistant", "content": "answer"}),
            response="answer",
            response_token_ids=(10, 11),
        ),
        verification=verification,
        reward=RewardResult(unshaped_reward=None, optimization_reward=0.0, token_rewards=(0.0, 0.0)),
        disposition=TrainingDisposition.train(),
        loss_mask=[1, 1],
        env_metrics={},
    )
    request = {
        "prompts": [[{"role": "user", "content": "q"}]],
        "env_classes": ["nemotron_ultra"],
        "env_extras": [{"extra_info": {"nemotron_ultra": {"agent": "genrm_simple_agent", "record_json": "{}"}}}],
        "sampling_params": None,
        "trajectory_ids": [TrajectoryID("prompt", 0)],
        "batch_metadata": BatchMetadata(global_step=0, training_phase="train"),
    }

    await runner._apply_genrm_cohort_rewards([output], request)

    runner.genrm_judge.generate_response.assert_not_called()
    assert output.verification is verification
    assert output.disposition.loss_eligible


def test_skipped_grading_warns_once_when_a_batch_has_no_ultra_rows(generator_cfg, tokenizer):
    skyrl_gym_cfg = DictConfig({"max_env_workers": 0, "nemotron_ultra": {"grading": "skip"}})
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, MagicMock(), tokenizer)
    messages = []
    sink_id = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        runner._warn_if_skip_has_no_ultra_rows({"env_classes": ["nemotron_ultra", "gsm8k"]})
        assert messages == []
        runner._warn_if_skip_has_no_ultra_rows({"env_classes": ["gsm8k", "gsm8k"]})
        runner._warn_if_skip_has_no_ultra_rows({"env_classes": ["gsm8k"]})
    finally:
        logger.remove(sink_id)

    assert len(messages) == 1
    assert "has no nemotron_ultra rows" in messages[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verification", "loss_eligible"),
    [
        (VerificationResult.skipped("grading is skipped"), True),
        (VerificationResult.unavailable("judge unreachable"), False),
    ],
)
async def test_agent_loop_trains_skipped_verdicts_and_masks_missing_ones(
    tokenizer, mock_llm, generator_cfg, skyrl_gym_cfg, use_env, verification, loss_eligible
):
    use_env(
        ScriptedEnv(
            BaseTextEnvStepOutput(observations=[], reward=0.0, done=True, metadata={}, verification=verification)
        )
    )
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, mock_llm, tokenizer)

    output = await runner.agent_loop(
        [{"role": "user", "content": "q"}], ENV_CLASS, {}, max_tokens=8, max_input_length=512
    )

    assert output.verification.status is verification.status
    assert output.disposition.loss_eligible is loss_eligible


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_forwards_environment_chat_options_and_structured_assistant_message(
    mock_make, tokenizer, mock_env, generator_cfg, skyrl_gym_cfg
):
    generator_cfg.use_conversation_multi_turn = True
    generator_cfg.require_exact_chat_transport = True
    generator_cfg.sampling_params.logprobs = 0
    tools = [{"type": "function", "name": "search", "parameters": {"type": "object"}}]
    mock_env.init.return_value = (
        [{"role": "user", "content": "look it up"}],
        {"chat_completion_params": {"tools": tools, "parallel_tool_calls": False}},
    )
    mock_env.step.side_effect = None
    mock_env.step.return_value = BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
    mock_make.return_value = mock_env
    assistant_message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"type": "function", "function": {"name": "search", "arguments": "{}"}}],
    }
    model_client = AsyncMock()
    model_client.generate.return_value = {
        "responses": ["<tool-call tokens>"],
        "response_ids": [[21, 22]],
        "prompt_ids": [[11, 12, 13]],
        "stop_reasons": ["tool_calls"],
        "response_logprobs": [[-0.1, -0.2]],
        "routed_experts": [np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8)],
        "prompt_logprobs": None,
        "assistant_messages": [assistant_message],
        "token_provenance": "engine",
    }
    runner = SkyRLGymTrajectoryRunner(
        trajectory_runner_cfg=generator_cfg,
        skyrl_gym_cfg=skyrl_gym_cfg,
        inference_engine_client=AsyncMock(),
        tokenizer=tokenizer,
        model_client=model_client,
    )

    output = await runner.agent_loop(
        [{"role": "user", "content": "look it up"}], ENV_CLASS, {}, max_tokens=8, max_input_length=512
    )

    request = model_client.generate.await_args.args[0]
    assert request["prompts"] == [[{"role": "user", "content": "look it up"}]]
    assert request["chat_completion_params"] == [{"tools": tools, "parallel_tool_calls": False}]
    evidence = mock_env.set_rollout_evidence.call_args.args[0]
    assert evidence.metadata["assistant_message"] == assistant_message
    assert output.evidence.prompt_token_ids == (11, 12, 13)
    assert output.evidence.response_token_ids == (21, 22)
    np.testing.assert_array_equal(output.evidence.routed_experts, [[[1, 2]], [[3, 4]]])


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_required_exact_chat_rejects_environment_without_chat_options(
    mock_make, tokenizer, mock_env, generator_cfg, skyrl_gym_cfg
):
    generator_cfg.require_exact_chat_transport = True
    mock_env.init.return_value = ([{"role": "user", "content": "look it up"}], {})
    mock_make.return_value = mock_env
    runner = SkyRLGymTrajectoryRunner(
        trajectory_runner_cfg=generator_cfg,
        skyrl_gym_cfg=skyrl_gym_cfg,
        inference_engine_client=AsyncMock(),
        tokenizer=tokenizer,
        model_client=AsyncMock(),
    )

    with pytest.raises(RuntimeError, match="did not provide chat_completion_params"):
        await runner.agent_loop(
            [{"role": "user", "content": "look it up"}], ENV_CLASS, {}, max_tokens=8, max_input_length=512
        )


def _structured_tool_turn_runner(mock_make, tokenizer, mock_env, generator_cfg, skyrl_gym_cfg, rendered_tool_ids):
    tools = [{"type": "function", "name": "python", "parameters": {"type": "object"}}]
    mock_env.init.return_value = (
        [{"role": "user", "content": "calculate"}],
        {"chat_completion_params": {"tools": tools}},
    )
    observation = {"role": "tool", "tool_call_id": "call-1", "content": "4"}
    mock_env.step.side_effect = [
        BaseTextEnvStepOutput(observations=[observation], reward=0.25, done=False, metadata={}),
        BaseTextEnvStepOutput(observations=[], reward=0.75, done=True, metadata={}),
    ]
    mock_make.return_value = mock_env
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "python", "arguments": "{}"}}],
    }
    model_client = AsyncMock()
    model_client.generate.side_effect = [
        {
            "responses": ["<tool>"],
            "response_ids": [[21, 22]],
            "prompt_ids": [[11, 12]],
            "stop_reasons": ["tool_calls"],
            "response_logprobs": [[-0.1, -0.2]],
            "assistant_messages": [tool_call],
            "token_provenance": "engine",
        },
        {
            "responses": ["four"],
            "response_ids": [[41]],
            "prompt_ids": [[11, 12, *rendered_tool_ids, 31, 32]],
            "stop_reasons": ["stop"],
            "response_logprobs": [[-0.3]],
            "assistant_messages": [{"role": "assistant", "content": "four", "tool_calls": []}],
            "token_provenance": "engine",
        },
    ]
    return SkyRLGymTrajectoryRunner(
        trajectory_runner_cfg=generator_cfg,
        skyrl_gym_cfg=skyrl_gym_cfg,
        inference_engine_client=AsyncMock(),
        tokenizer=tokenizer,
        model_client=model_client,
    )


@pytest.mark.asyncio
@patch("skyrl_gym.make")
@pytest.mark.parametrize(
    (
        "rendered_tool_ids",
        "expected_response_ids",
        "expected_mask",
        "expected_logprobs",
        "expected_token_rewards",
        "expected_provenance",
    ),
    [
        (
            [21, 22],
            [21, 22, 31, 32, 41],
            [1, 1, 0, 0, 1, 0],
            [-0.1, -0.2, 0.0, 0.0, -0.3, 0.0],
            [0.0, 0.25, 0.0, 0.0, 0.75, 0.0],
            TokenProvenance.ENGINE,
        ),
        (
            [23, 24],
            [23, 24, 31, 32, 41],
            [0, 0, 0, 0, 1, 0],
            [0.0, 0.0, 0.0, 0.0, -0.3, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            TokenProvenance.RECONSTRUCTED,
        ),
    ],
    ids=["exact-prefix", "canonicalized-prefix"],
)
async def test_agent_loop_handles_backend_rendered_prefix_across_structured_tool_turns(
    mock_make,
    tokenizer,
    mock_env,
    generator_cfg,
    skyrl_gym_cfg,
    rendered_tool_ids,
    expected_response_ids,
    expected_mask,
    expected_logprobs,
    expected_token_rewards,
    expected_provenance,
):
    generator_cfg.use_conversation_multi_turn = False
    generator_cfg.sampling_params.logprobs = 0
    runner = _structured_tool_turn_runner(
        mock_make, tokenizer, mock_env, generator_cfg, skyrl_gym_cfg, rendered_tool_ids
    )
    model_client = runner.model_client
    tool_call = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "python", "arguments": "{}"}}],
    }
    observation = {"role": "tool", "tool_call_id": "call-1", "content": "4"}

    output = await runner.agent_loop(
        [{"role": "user", "content": "calculate"}], ENV_CLASS, {}, max_tokens=8, max_input_length=512
    )

    second_request = model_client.generate.await_args_list[1].args[0]
    assert second_request["prompts"] == [[{"role": "user", "content": "calculate"}, tool_call, observation]]
    assert second_request["chat_continuations"] == [
        {"served_prefix_token_ids": [11, 12, 21, 22], "assistant_message_index": 1}
    ]
    assert output.evidence.prompt_token_ids == (11, 12)
    assert output.evidence.response_token_ids == (*expected_response_ids, EOS)
    assert output.loss_mask == expected_mask
    np.testing.assert_allclose(output.evidence.behavior_logprobs, expected_logprobs)
    assert output.reward.token_rewards == pytest.approx(expected_token_rewards)
    assert output.token_provenance == expected_provenance
    assert output.reward.optimization_reward == 1.0


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_agent_loop_required_exact_chat_rejects_canonicalized_prefix(
    mock_make, tokenizer, mock_env, generator_cfg, skyrl_gym_cfg
):
    generator_cfg.use_conversation_multi_turn = False
    generator_cfg.require_exact_chat_transport = True
    generator_cfg.sampling_params.logprobs = 0
    runner = _structured_tool_turn_runner(mock_make, tokenizer, mock_env, generator_cfg, skyrl_gym_cfg, [23, 24])

    with pytest.raises(ExactChatTransportError, match="did not preserve the served token prefix"):
        await runner.agent_loop(
            [{"role": "user", "content": "calculate"}], ENV_CLASS, {}, max_tokens=8, max_input_length=512
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "response_ids", "response_logprobs", "expected_ids", "expected_mask", "expected_rewards"),
    [
        ("stop", [10, 11], [-0.1, -0.2], [10, 11, EOS], [1, 1, 0], [0.0, 1.0, 0.0]),
        ("stop", [10, 11, EOS], [-0.1, -0.2, -0.3], [10, 11, EOS], [1, 1, 1], [0.0, 0.0, 1.0]),
        ("length", [10, 11], [-0.1, -0.2], [10, 11], [1, 1], [0.0, 1.0]),
    ],
    ids=["synthetic-eos", "sampled-eos", "length"],
)
async def test_terminal_assembly_masks_unsampled_tokens(
    tokenizer,
    generator_cfg,
    skyrl_gym_cfg,
    use_env,
    stop_reason,
    response_ids,
    response_logprobs,
    expected_ids,
    expected_mask,
    expected_rewards,
):
    generator_cfg.sampling_params.logprobs = 0
    generator_cfg.use_conversation_multi_turn = False
    use_env(ScriptedEnv(BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})))
    engine = engine_returning(
        {
            "responses": ["answer"],
            "stop_reasons": [stop_reason],
            "response_ids": [response_ids],
            "response_logprobs": [response_logprobs],
        }
    )
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, engine, tokenizer)

    output = await runner.agent_loop(
        [{"role": "user", "content": "Question"}], ENV_CLASS, {}, max_tokens=8, max_input_length=512
    )

    assert list(output.evidence.response_token_ids) == expected_ids
    assert output.loss_mask == expected_mask
    assert output.reward.token_rewards == tuple(expected_rewards)
    assert output.reward.optimization_reward == 1.0
    assert output.evidence.behavior_logprobs is not None
    assert len(output.evidence.behavior_logprobs) == len(expected_ids)
    sampled_trainable_logprobs = [
        logprob
        for logprob, trainable in zip(output.evidence.behavior_logprobs, output.loss_mask, strict=True)
        if trainable
    ]
    assert sampled_trainable_logprobs == response_logprobs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stop_reason", "response_ids", "expected_mask"),
    [("length", [10, 11, 12], [0, 0, 0]), ("stop", [10, 11, EOS], [1, 1, 1])],
    ids=["truncated", "complete"],
)
async def test_overlong_filtering_masks_only_truncated_trajectories(
    tokenizer, generator_cfg, skyrl_gym_cfg, use_env, stop_reason, response_ids, expected_mask
):
    generator_cfg.apply_overlong_filtering = True
    generator_cfg.use_conversation_multi_turn = False
    use_env(ScriptedEnv(BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})))
    engine = engine_returning({"responses": ["answer"], "stop_reasons": [stop_reason], "response_ids": [response_ids]})
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, engine, tokenizer)

    output = await runner.run(single_prompt_request())

    assert output["response_ids"] == [response_ids]
    assert output["loss_masks"] == [expected_mask]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_conversation_multi_turn", [True, False], ids=["chat-turns", "single-message"])
async def test_multi_turn_assembly_aligns_per_token_fields_across_observations(
    tokenizer, generator_cfg, skyrl_gym_cfg, use_env, use_conversation_multi_turn
):
    """Sampled-token fields keep their positions around masked observation tokens, and each step's reward lands
    on the last sampled token of its turn."""
    generator_cfg.sampling_params.logprobs = 2
    generator_cfg.use_conversation_multi_turn = use_conversation_multi_turn
    generator_cfg.max_turns = 2
    use_env(
        ScriptedEnv(
            BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": "tool result"}], reward=0.3, done=False, metadata={}
            ),
            BaseTextEnvStepOutput(observations=[], reward=1.7, done=True, metadata={}),
        )
    )
    engine = engine_returning(
        {
            "responses": ["first"],
            "stop_reasons": ["stop"],
            "response_ids": [[10, EOS]],
            "response_logprobs": [[-0.1, -0.2]],
            "routed_experts": [np.asarray([[[1, 2]], [[3, 4]]], dtype=np.uint8)],
            "student_topk_indices": [[[11, 12], [13, 14]]],
            "behavior_topk_logprobs": [[[-0.1, -2.0], [-0.2, -1.9]]],
        },
        {
            "responses": ["second"],
            "stop_reasons": ["stop"],
            "response_ids": [[20, EOS]],
            "response_logprobs": [[-0.3, -0.4]],
            "routed_experts": [np.asarray([[[5, 6]], [[7, 8]]], dtype=np.uint8)],
            "student_topk_indices": [[[21, 22], [23, 24]]],
            "behavior_topk_logprobs": [[[-0.3, -1.8], [-0.4, -1.7]]],
        },
    )
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, engine, tokenizer)

    output = await runner.run(single_prompt_request())

    if use_conversation_multi_turn:
        # The observation is a new chat turn that also opens the next assistant turn.
        observation_ids = tokenizer.encode(
            "\n<|im_start|>user\ntool result<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False
        )
        first_turn = 2
    else:
        # The observation continues the same assistant message, so the first turn's EOS is dropped.
        observation_ids = tokenizer.encode("tool result", add_special_tokens=False)
        first_turn = 1
    gap = len(observation_ids)
    assert output["response_ids"] == [[10, EOS][:first_turn] + observation_ids + [20, EOS]]
    assert output["loss_masks"] == [[1] * first_turn + [0] * gap + [1, 1]]
    np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.1, -0.2][:first_turn] + [0.0] * gap + [-0.3, -0.4])
    expected_rewards = [0.0] * (first_turn + gap + 2)
    expected_rewards[first_turn - 1] = 0.3
    expected_rewards[-1] = 1.7
    assert output["rewards"] == [pytest.approx(expected_rewards)]
    if not use_conversation_multi_turn:
        # Dropping the first turn's EOS leaves no aligned top-K capture for the single-message format.
        return

    np.testing.assert_array_equal(
        output["rollout_routed_experts"][0],
        [[[1, 2]], [[3, 4]]] + [[[0, 0]]] * gap + [[[5, 6]], [[7, 8]]],
    )
    np.testing.assert_array_equal(
        output["student_topk_indices"][0], [[11, 12], [13, 14]] + [[-1, -1]] * gap + [[21, 22], [23, 24]]
    )
    np.testing.assert_allclose(
        output["behavior_topk_logprobs"][0],
        [[-0.1, -2.0], [-0.2, -1.9]] + [[0.0, 0.0]] * gap + [[-0.3, -1.8], [-0.4, -1.7]],
    )
    behavior = torch.from_numpy(np.stack(output["rollout_logprobs"]))
    ratios = torch.tensor([[1.5, 4.0] + [torch.nan] * gap + [1.0, 0.5]])
    correction = compute_correction(
        behavior + ratios.log(), behavior, torch.tensor(output["loss_masks"]), load_correction("tis")
    )
    torch.testing.assert_close(correction.weights, torch.tensor([[1.5, 2.0] + [0.0] * gap + [1.0, 0.5]]))
    assert correction.metrics["policy/correction/weight_mean"] == pytest.approx(1.25)
    assert correction.metrics["policy/correction/truncated_fraction"] == pytest.approx(0.25)
    output["trajectory_ids"] = [TrajectoryID("tool-trajectory", 0)]
    work = build_teacher_scoring_work(
        output,
        route_ids=("math",),
        teacher_id="math-teacher",
        tokenizer_fingerprint="same-tokenizer",
        plan_version="test-plan",
        coefficient=1.0,
        route_weights=(1.0,),
        evidence=TeacherEvidenceKind.STUDENT_SELECTED_TOPK,
        top_k=2,
    )
    assert work.request.student_selected_mask.tolist() == [[True, True] + [False] * gap + [True, True]]
    assert np.isnan(work.request.behavior_topk_logprobs[0, 2 : 2 + gap].numpy()).all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("postprocessed_action", "keeps_logprobs"),
    [("answer", True), ("rewritten answer", False)],
    ids=["same-tokens", "rewritten"],
)
async def test_postprocessed_action_replaces_response_and_keeps_only_aligned_logprobs(
    tokenizer, generator_cfg, skyrl_gym_cfg, use_env, postprocessed_action, keeps_logprobs
):
    generator_cfg.sampling_params.logprobs = 0
    generator_cfg.use_conversation_multi_turn = False
    sampled_ids = tokenizer.encode("answer", add_special_tokens=False)
    use_env(
        ScriptedEnv(
            BaseTextEnvStepOutput(
                observations=[], reward=1.0, done=True, metadata={}, postprocessed_action=postprocessed_action
            )
        )
    )
    engine = engine_returning(
        {
            "responses": ["answer"],
            "stop_reasons": ["stop"],
            "response_ids": [sampled_ids],
            "response_logprobs": [[-0.1] * len(sampled_ids)],
        }
    )
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, engine, tokenizer)

    output = await runner.run(single_prompt_request())

    assert output["response_ids"] == [tokenizer.encode(postprocessed_action, add_special_tokens=False) + [EOS]]
    if keeps_logprobs:
        np.testing.assert_allclose(output["rollout_logprobs"][0], [-0.1] * len(sampled_ids) + [0.0])
    else:
        assert output["rollout_logprobs"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("extra_input_budget", "expect_multiple_turns"), [(0, False), (60, True)], ids=["first-turn", "later-turn"]
)
async def test_input_length_limit_ends_conversation_at_last_completed_turn(
    tokenizer, generator_cfg, skyrl_gym_cfg, use_env, extra_input_budget, expect_multiple_turns
):
    generator_cfg.use_conversation_multi_turn = True
    generator_cfg.max_turns = 50
    env = use_env(
        ScriptedEnv(
            BaseTextEnvStepOutput(
                observations=[{"role": "user", "content": "next"}], reward=0.5, done=False, metadata={}
            )
        )
    )
    engine = MagicMock()
    engine.generate = AsyncMock(
        return_value={"responses": ["step"], "stop_reasons": ["stop"], "response_ids": [[10, 11, 12, EOS]]}
    )
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, engine, tokenizer)
    prompt = [{"role": "user", "content": "Start"}]
    prompt_length = len(normalize_token_ids(tokenizer.apply_chat_template(prompt, add_generation_prompt=True)))

    output = await runner.agent_loop(
        prompt, ENV_CLASS, {}, max_tokens=100, max_input_length=prompt_length + extra_input_budget
    )

    response_ids = list(output.evidence.response_token_ids)
    assert output.evidence.stop_reason == "length"
    assert (env.num_steps > 1) is expect_multiple_turns
    # The observation that pushed the input over the limit is not part of the response.
    assert response_ids[-4:] == [10, 11, 12, EOS]
    assert output.loss_mask[-4:] == [1, 1, 1, 1]
    assert len(output.loss_mask) == len(response_ids)
    assert output.reward.optimization_reward == pytest.approx(0.5 * env.num_steps)
    assert output.reward.token_rewards.count(0.5) == env.num_steps


@pytest.mark.asyncio
@patch("skyrl_gym.make")
@pytest.mark.parametrize("retokenize_chat_history", [False, True])
async def test_agent_loop_initial_prompt_over_budget_returns_empty_rollout(
    mock_make, tokenizer, mock_llm, mock_env, generator_cfg, skyrl_gym_cfg, retokenize_chat_history
):
    generator_cfg.use_conversation_multi_turn = retokenize_chat_history
    if retokenize_chat_history:
        generator_cfg.chat_template = {"source": "name", "name_or_path": "qwen3_without_thinking"}
    mock_make.return_value = mock_env
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})
    mock_env.step.side_effect = AssertionError("an overlong initial prompt must not enter the environment")
    mock_llm.generate.side_effect = AssertionError("an overlong initial prompt must not reach inference")
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, mock_llm, tokenizer)
    max_input_length = 4

    output = await runner.agent_loop(
        [{"role": "user", "content": "Initial input"}], ENV_CLASS, {}, max_tokens=8, max_input_length=max_input_length
    )

    assert len(output.evidence.prompt_token_ids) > max_input_length
    assert list(output.evidence.response_token_ids) == []
    assert output.loss_mask == []
    assert output.evidence.behavior_logprobs is not None and output.evidence.behavior_logprobs.size == 0
    assert output.reward.optimization_reward == 0.0
    assert output.reward.token_rewards == (None if retokenize_chat_history else ())
    assert output.evidence.stop_reason == "length"


@pytest.mark.asyncio
async def test_generate_aggregates_aime_step_metadata(tokenizer, mock_llm, generator_cfg):
    mock_llm.generate = AsyncMock(
        return_value={
            "responses": ["Answer: \\boxed{42}"],
            "stop_reasons": ["stop"],
            "response_ids": [SAMPLED_IDS.copy()],
        }
    )
    runner = SkyRLGymTrajectoryRunner(
        trajectory_runner_cfg=generator_cfg,
        skyrl_gym_cfg=DictConfig({"max_env_workers": 0, "aime": {"evaluation_token_budget": 8}}),
        inference_engine_client=mock_llm,
        tokenizer=tokenizer,
    )

    trajectory_batch = await runner.run(
        {
            "prompts": [[{"role": "user", "content": "Solve the problem"}]],
            "env_extras": [{"reward_model": {"ground_truth": "42"}}],
            "env_classes": ["aime"],
        }
    )

    assert trajectory_batch["unshaped_rewards"] == [1.0]
    assert trajectory_batch["rollout_metrics"]["environment/acc"] == pytest.approx(1.0)
    assert trajectory_batch["rollout_metrics"][
        "environment/answered_within_evaluation_budget_fraction"
    ] == pytest.approx(1.0)


class _InlineExecutor(Executor):
    """Run each environment call as it is submitted, through the runner's executor path."""

    def submit(self, fn, /, *args, **kwargs):
        future = Future()
        future.set_result(fn(*args, **kwargs))
        return future


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_a_rollout_call_publishes_its_phases_and_waits(
    mock_make, tokenizer, mock_llm, mock_env, generator_cfg, skyrl_gym_cfg, delivered_telemetry
):
    generator_cfg.use_conversation_multi_turn = False
    mock_make.return_value = mock_env
    mock_env.init.return_value = ([{"role": "user", "content": "Initial input"}], {})
    runner = SkyRLGymTrajectoryRunner(generator_cfg, skyrl_gym_cfg, mock_llm, tokenizer)
    runner.env_executor = _InlineExecutor()

    with observe_rollout_call(step=3, mode="async", enabled=True):
        await runner.run(single_prompt_request("2 + 2?"))

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
    # One model call; the environment's init, step and close each take the executor path.
    assert waits == {"model_client_await": 1, "env_await": 3, "env_queue": 3, "env_exec": 3, "env_resume": 3}
    (call,) = delivered_telemetry.select("rollout_call", step="3")
    assert call["attributes"]["outcome"] == "success"


@pytest.mark.asyncio
@pytest.mark.parametrize("judge_fails", [False, True])
async def test_genrm_failed_rollouts_keep_their_failure_and_never_enter_comparisons(
    generator_cfg, tokenizer, judge_fails
):
    runner = SkyRLGymTrajectoryRunner(
        generator_cfg,
        DictConfig(
            {
                "max_env_workers": 0,
                "nemotron_ultra": {
                    "genrm": {
                        "num_rollouts_per_prompt": 3,
                        "group_answer_length_penalty_coeff": 0.0,
                        "genrm_parse_retries": 0,
                        "reasoning_bonus": 0.0,
                        "answer_bonus": 0.0,
                    },
                },
            }
        ),
        MagicMock(),
        tokenizer,
    )
    compared = []

    class Judge:
        def generate_response(self, messages, *, metadata, **kwargs):
            compared.extend([metadata["response_1"], metadata["response_2"]])
            if judge_fails:
                raise ConnectionError("judge unavailable")
            return '{"score_1":4,"score_2":4,"ranking":3.5}'

    runner.genrm_judge = Judge()
    outputs = [
        AgentLoopOutput(
            evidence=RolloutEvidence(
                messages=({"role": "assistant", "content": answer},), response=answer, response_token_ids=(10, 11)
            ),
            verification=VerificationResult.verified(3.0),
            reward=RewardResult(unshaped_reward=3.0, optimization_reward=3.0, token_rewards=(0.0, 3.0)),
            disposition=TrainingDisposition.train(),
            loss_mask=[1, 1],
            env_metrics={},
        )
        for answer in ("valid-a", "failed", "valid-b")
    ]
    original_failure = VerificationResult.error("generation failed", diagnostics={"exception_type": "RuntimeError"})
    outputs[1].verification = original_failure
    outputs[1].disposition = TrainingDisposition.mask("generation failed")
    outputs[1].reward = RewardResult(unshaped_reward=None, optimization_reward=0.0)
    extras = {
        "extra_info": {"nemotron_ultra": {"agent": "genrm_simple_agent", "record_json": '{"principle":"correct"}'}}
    }
    request = {
        "prompts": [[{"role": "user", "content": "q"}]] * 3,
        "env_extras": [extras] * 3,
        "trajectory_ids": [TrajectoryID("same-prompt", i) for i in range(3)],
        "batch_metadata": None,
    }
    await runner._apply_genrm_cohort_rewards(outputs, request)
    assert "failed" not in compared
    assert outputs[1].verification is original_failure
    assert not outputs[1].disposition.loss_eligible
    if judge_fails:
        assert all(not output.disposition.loss_eligible for output in outputs)
        assert all(output.reward.optimization_reward == 0.0 for output in outputs)
        assert outputs[0].verification.status.value == "error"
    else:
        assert outputs[0].reward.optimization_reward == 4.0
        assert outputs[2].reward.optimization_reward == 4.0


@pytest.mark.asyncio
@pytest.mark.parametrize("custom_template", [False, True], ids=["default-chat", "configured-token-path"])
async def test_cat_count_preserves_sampled_evidence_and_verification(
    generator_cfg, skyrl_gym_cfg, tokenizer, custom_template
):
    generator_cfg.use_conversation_multi_turn = not custom_template
    generator_cfg.sampling_params.logprobs = 0
    prompt = [{"role": "user", "content": "Reply with the word cat exactly 2 times."}]
    if custom_template:
        generator_cfg.chat_template.name_or_path = "qwen2_5_with_generation_tag_simplified"
        prompt_ids = tokenizer.encode(
            "<|im_start|>user\nReply with the word cat exactly 2 times.<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
    else:
        prompt_ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_dict=False)
    model_client = AsyncMock()
    model_client.generate.return_value = {
        "responses": ["cat cat"],
        "response_ids": [[21, 22, EOS]],
        "stop_reasons": ["stop"],
        "response_logprobs": [[-0.1, -0.2, -0.3]],
        "token_provenance": "engine",
    }
    runner = SkyRLGymTrajectoryRunner(
        generator_cfg,
        skyrl_gym_cfg,
        AsyncMock(),
        tokenizer,
        model_client=model_client,
    )
    batch = await runner.run(
        {
            "prompts": [prompt],
            "env_extras": [{"extra_info": {"n": 2}}],
            "env_classes": ["cat_count"],
        }
    )

    assert batch["prompt_token_ids"] == [prompt_ids]
    assert batch["response_ids"] == [[21, 22, EOS]]
    np.testing.assert_allclose(batch["rollout_logprobs"][0], [-0.1, -0.2, -0.3])
    assert batch["rewards"] == [[0.0, 0.0, 1.0]]
    assert batch["loss_masks"] == [[1, 1, 1]]
    assert batch["verification_results"][0].passed is True
    assert batch["env_metrics"][0]["exact_n2"] == 1.0
