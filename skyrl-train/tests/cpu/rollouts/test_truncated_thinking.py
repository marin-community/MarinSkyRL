"""A final reply that never closes its reasoning block submits no answer."""

from unittest.mock import AsyncMock

import pytest
from taskcompendium.grading_result import Outcome

from skyrl_train.evaluate import _calculate_eval_metrics
from skyrl_train.metric_names import TASK_ROLLOUT_METRIC_PREFIX
from skyrl_train.rollouts.task_projections import StepTaskProjection, WholeTaskProjection
from skyrl_train.rollouts.task_worker import TaskRolloutWorker
from skyrl_train.trajectory_runners.model_clients import TRUNCATED_THINKING_KEY, DirectModelClient
from tests.cpu.rollouts.engine_fakes import InferenceClient, Tokenizer

GENERATION_PROMPT = "<|im_start|>assistant\n<think>\n"


class TextTokenizer:
    """Decode each token id to a fixed piece of text: 1-2 are the prompt, 3 onward the response."""

    eos_token_id = 99

    def __init__(self, response: str):
        self.pieces = {1: "<|im_start|>user\nQ<|im_end|>\n", 2: GENERATION_PROMPT, 3: response}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.pieces.get(token, "") for token in ids)


async def _chat(response: str, message: dict) -> dict:
    engine = AsyncMock()
    engine.model_name = "qwen3-thinking-2507"
    engine.tokenizer = TextTokenizer(response)
    engine.tokenize.return_value = {"tokens": [1, 2]}
    engine.chat_completion.return_value = {
        "choices": [{"message": message, "finish_reason": "length", "token_ids": [3], "logprobs": None}]
    }
    return await DirectModelClient(engine).generate(
        {"prompts": [[{"role": "user", "content": "Q"}]], "chat_completion_params": [{}]}
    )


@pytest.mark.asyncio
async def test_reply_cut_off_inside_template_opened_reasoning_has_no_answer():
    reasoning = "The total is 42, though I should check the units"

    result = await _chat(reasoning, {"role": "assistant", "content": reasoning})

    assert result["assistant_messages"][0]["content"] == ""
    assert result["responses"] == [""]
    assert result[TRUNCATED_THINKING_KEY] == [True]
    assert result["response_ids"] == [[3]]


@pytest.mark.asyncio
async def test_closed_reasoning_keeps_the_final_answer():
    response = "The total is 42.\n</think>\n\nTotal: 42 J"

    result = await _chat(response, {"role": "assistant", "content": response})

    assert result["assistant_messages"][0]["content"] == "Total: 42 J"
    assert result[TRUNCATED_THINKING_KEY] == [False]


@pytest.mark.asyncio
async def test_tool_call_turn_inside_open_reasoning_is_unchanged():
    message = {
        "role": "assistant",
        "content": "Check the file",
        "tool_calls": [{"id": "c", "type": "function", "function": {"name": "shell", "arguments": "{}"}}],
    }

    result = await _chat("Check the file", message)

    assert result["assistant_messages"][0] == message
    assert result[TRUNCATED_THINKING_KEY] == [False]


@pytest.mark.asyncio
@pytest.mark.parametrize("projection_type", [WholeTaskProjection, StepTaskProjection])
@pytest.mark.parametrize("truncated", [True, False])
async def test_truncated_final_reply_fails_submission_and_is_counted(task_inputs, projection_type, truncated):
    class ThinkingClient(InferenceClient):
        async def generate(self, request):
            output = await super().generate(request)
            if truncated:
                output["assistant_messages"] = [{"role": "assistant", "content": ""}]
            output[TRUNCATED_THINKING_KEY] = [truncated]
            return output

    config, request = task_inputs
    worker = TaskRolloutWorker(config, projection_type(config, Tokenizer()), ThinkingClient(), {}, shutdown_timeout=30)
    try:
        (rollout,) = await worker.generate(request)
        batch = await worker.training_batch(request, [rollout])
    finally:
        await worker.shutdown()

    assert rollout.grade.status is (Outcome.SUBMISSION_FAILURE if truncated else Outcome.GRADED)
    assert rollout.grade.reward == (0.0 if truncated else 1.0)
    assert batch["rollout_metrics"][f"{TASK_ROLLOUT_METRIC_PREFIX}{TRUNCATED_THINKING_KEY}"] == float(truncated)
    eval_metrics = _calculate_eval_metrics(batch, [rollout.task_id], ["fixture"], 1)
    assert eval_metrics[f"eval/all/{TRUNCATED_THINKING_KEY}"] == float(truncated)
