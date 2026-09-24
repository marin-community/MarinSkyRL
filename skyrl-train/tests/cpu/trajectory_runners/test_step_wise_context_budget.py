"""Regression coverage for request-level context clamping in step-wise rollouts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import DictConfig, open_dict

from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.config.utils import get_default_config
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection
from skyrl_train.trajectory_runners.step_wise import StepWiseRolloutCollector
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner, TrajectoryPipeline
from skyrl_train.trajectory_runners.step_wise import clamp_generation_tokens


class _RecordingInferenceEngine:
    def __init__(self, response_logprobs=None, topk=None):
        self.requests = []
        self.response_logprobs = response_logprobs
        self.topk = topk

    async def generate(self, request):
        self.requests.append(request)
        output = {
            "responses": ["ok"],
            "response_ids": [[7, 8]],
            "stop_reasons": ["stop"],
            "response_logprobs": [self.response_logprobs],
        }
        if self.topk is not None:
            output["student_topk_indices"] = [self.topk[0]]
            output["behavior_topk_logprobs"] = [self.topk[1]]
        return output


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = [1, 2, 3, 4]
    tokenizer.eos_token_id = 4
    tokenizer.eos_token = "<eos>"
    return tokenizer


def test_clamp_generation_tokens_reserves_only_remaining_request_window():
    assert clamp_generation_tokens(prompt_tokens=30, request_window_tokens=32, requested_output_tokens=16) == 2
    assert clamp_generation_tokens(prompt_tokens=32, request_window_tokens=32, requested_output_tokens=16) == 0


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_step_wise_generation_clamps_final_request_to_tokenized_window(mock_make):
    cfg = get_default_config().generator
    cfg.batched = False
    cfg.use_conversation_multi_turn = True
    cfg.max_turns = 2
    cfg.max_input_length = 4
    cfg.sampling_params.max_generate_length = 16
    with open_dict(cfg.engine_init_kwargs):
        cfg.engine_init_kwargs.max_model_len = 6
    cfg.chat_template_kwargs = {}

    environment = MagicMock()
    environment.init.return_value = ([{"role": "user", "content": "task"}], {})
    environment.step.return_value = BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
    environment.get_metrics.return_value = {}
    mock_make.return_value = environment
    engine = _RecordingInferenceEngine()
    tokenizer = _tokenizer()
    runner = SkyRLGymTrajectoryRunner(
        cfg,
        DictConfig({"max_env_workers": 0}),
        engine,
        tokenizer,
        pipeline=TrajectoryPipeline(StepWiseRolloutCollector, StepWiseTrajectoryProjection(cfg, tokenizer)),
    )
    collector = StepWiseRolloutCollector(runner)

    outputs = await collector.agent_loop(
        [{"role": "user", "content": "task"}],
        "test_env",
        {},
        max_tokens=16,
        max_input_length=4,
    )

    assert outputs[0].evidence.response_token_ids == (7, 8)
    assert engine.requests[0]["sampling_params"]["max_tokens"] == 2


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_step_wise_stop_eos_keeps_published_behavior_evidence_aligned(mock_make):
    cfg = get_default_config().generator
    cfg.batched = False
    cfg.use_conversation_multi_turn = True
    cfg.max_turns = 2
    cfg.sampling_params.stop = ["ok"]
    cfg.append_eos_token_after_stop_str_in_multi_turn = True
    cfg.chat_template_kwargs = {}

    environment = MagicMock()
    environment.init.return_value = ([{"role": "user", "content": "task"}], {})
    environment.step.return_value = BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
    environment.get_metrics.return_value = {}
    mock_make.return_value = environment
    tokenizer = _tokenizer()
    runner = SkyRLGymTrajectoryRunner(
        cfg,
        DictConfig({"max_env_workers": 0}),
        _RecordingInferenceEngine(response_logprobs=[-0.1, -0.2]),
        tokenizer,
        pipeline=TrajectoryPipeline(StepWiseRolloutCollector, StepWiseTrajectoryProjection(cfg, tokenizer)),
    )

    outputs = await StepWiseRolloutCollector(runner).agent_loop(
        [{"role": "user", "content": "task"}],
        "test_env",
        {},
        max_tokens=16,
        max_input_length=4,
    )

    published = environment.set_rollout_evidence.call_args.args[0]
    assert published.response_token_ids == (7, 8, tokenizer.eos_token_id)
    assert published.behavior_logprobs == (-0.1, -0.2, 0.0)
    assert outputs[0].evidence.behavior_logprobs == (-0.1, -0.2, 0.0)


@pytest.mark.asyncio
@patch("skyrl_gym.make")
async def test_step_wise_collector_preserves_student_topk(mock_make):
    cfg = get_default_config().generator
    cfg.batched = False
    cfg.use_conversation_multi_turn = True
    cfg.max_turns = 1
    cfg.sampling_params.logprobs = 2
    cfg.chat_template_kwargs = {}
    environment = MagicMock()
    environment.init.return_value = ([{"role": "user", "content": "task"}], {})
    environment.step.return_value = BaseTextEnvStepOutput(observations=[], reward=1.0, done=True, metadata={})
    environment.get_metrics.return_value = {}
    mock_make.return_value = environment
    tokenizer = _tokenizer()
    engine = _RecordingInferenceEngine(
        response_logprobs=[-0.1, -0.2],
        topk=([[7, 9], [8, 10]], [[-0.1, -1.1], [-0.2, -1.2]]),
    )
    runner = SkyRLGymTrajectoryRunner(
        cfg,
        DictConfig({"max_env_workers": 0}),
        engine,
        tokenizer,
        pipeline=TrajectoryPipeline(StepWiseRolloutCollector, StepWiseTrajectoryProjection(cfg, tokenizer)),
    )

    outputs = await StepWiseRolloutCollector(runner).agent_loop(
        [{"role": "user", "content": "task"}], "test_env", {}, max_tokens=16, max_input_length=4
    )

    assert outputs[0].evidence.student_topk_indices == ((7, 9), (8, 10))
    assert outputs[0].evidence.behavior_topk_logprobs == ((-0.1, -1.1), (-0.2, -1.2))
