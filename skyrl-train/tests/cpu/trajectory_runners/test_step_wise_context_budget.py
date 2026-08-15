"""Regression coverage for request-level context clamping in step-wise rollouts."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from omegaconf import open_dict

from skyrl_gym.envs.base_text_env import BaseTextEnvStepOutput
from skyrl_train.config.utils import get_default_config
from skyrl_train.trajectory_runners.projections import StepWiseTrajectoryProjection
from skyrl_train.trajectory_runners.step_wise import StepWiseRolloutCollector
from skyrl_train.trajectory_runners.skyrl_gym import SkyRLGymTrajectoryRunner, TrajectoryPipeline
from skyrl_train.trajectory_runners.step_wise import clamp_generation_tokens


class _RecordingInferenceEngine:
    def __init__(self):
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return {
            "responses": ["ok"],
            "response_ids": [[7, 8]],
            "stop_reasons": ["stop"],
            "response_logprobs": [None],
        }


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
        MagicMock(max_env_workers=0),
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

    assert outputs[0].response_ids == [7, 8]
    assert engine.requests[0]["sampling_params"]["max_tokens"] == 2
